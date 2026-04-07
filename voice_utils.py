"""Shared voice conversion utilities used by app.py and translate.py."""

import contextlib
import math
import threading
import time
import torch

SAMPLE_RATE = 16000
CHUNK_SECONDS = 15       # CPU default — WavLM attention is O(N²)
CHUNK_SECONDS_GPU = 5    # GPU needs smaller chunks: WavLM attention allocates ~8 GB for 15s on CUDA


def extract_features_chunked(
    knn_vc,
    wav: torch.Tensor,
    chunk_seconds: int | None = None,
    progress_cb=None,
) -> torch.Tensor:
    """Extract WavLM features in chunks to avoid O(N²) attention hang on long audio."""
    if chunk_seconds is None:
        chunk_seconds = CHUNK_SECONDS_GPU if wav.device.type == "cuda" else CHUNK_SECONDS
    chunk_samples = chunk_seconds * SAMPLE_RATE
    total_samples = wav.shape[1]

    if total_samples <= chunk_samples:
        return knn_vc.get_features(wav)

    chunks = []
    n_chunks = (total_samples + chunk_samples - 1) // chunk_samples
    is_cuda = wav.device.type == "cuda"
    for i, start in enumerate(range(0, total_samples, chunk_samples)):
        end = min(start + chunk_samples, total_samples)
        feats = knn_vc.get_features(wav[:, start:end])
        chunks.append(feats.cpu() if is_cuda else feats)
        if is_cuda and (i + 1) % 10 == 0:
            torch.cuda.empty_cache()
        msg = f"Features: chunk {i+1}/{n_chunks} ({end//SAMPLE_RATE}s / {total_samples//SAMPLE_RATE}s)"
        print(msg)
        if progress_cb:
            progress_cb(msg)

    result = torch.cat(chunks, dim=0)
    # For GPU: keep result on CPU — kNN lookup in match() runs on CPU anyway
    return result if not is_cuda else result  # already CPU from chunks.append(feats.cpu())


def get_matching_set_chunked(
    knn_vc,
    wav: torch.Tensor,
    chunk_seconds: int | None = None,
    progress_cb=None,
) -> torch.Tensor:
    """Chunked replacement for knn_vc.get_matching_set([wav]).

    knn_vc.get_matching_set runs WavLM on the full reference audio in one pass —
    OOM for long reference files on GPU. This version chunks the reference the
    same way as extract_features_chunked and returns a CPU tensor (same as the
    original, since kNN lookup runs on CPU).
    """
    if chunk_seconds is None:
        chunk_seconds = CHUNK_SECONDS_GPU if wav.device.type == "cuda" else CHUNK_SECONDS
    chunk_samples = chunk_seconds * SAMPLE_RATE
    total_samples = wav.shape[1]

    if total_samples <= chunk_samples:
        return knn_vc.get_matching_set([wav])   # short ref: use original (applies VAD/weights)

    msg = f"Reference audio is long ({total_samples//SAMPLE_RATE}s) — chunking into {chunk_seconds}s pieces"
    print(msg)
    if progress_cb:
        progress_cb(msg)

    is_cuda = wav.device.type == "cuda"
    chunks = []
    n_chunks = (total_samples + chunk_samples - 1) // chunk_samples
    for i, start in enumerate(range(0, total_samples, chunk_samples)):
        end = min(start + chunk_samples, total_samples)
        feats = knn_vc.get_features(wav[:, start:end])
        chunks.append(feats.cpu() if is_cuda else feats)
        if is_cuda:
            torch.cuda.empty_cache()

    return torch.cat(chunks, dim=0).cpu()


# WavLM frame hop: 50 fps at 16 kHz → 320 samples per frame
_WAVLM_HOP = 320
# Default chunk: ~20 seconds of audio per HiFiGAN call (safe for 12 GB VRAM)
MATCH_CHUNK_FRAMES = 1000
# Overlap added on each side so HiFiGAN has receptive-field context at boundaries
MATCH_OVERLAP_FRAMES = 32
# Samples to crossfade at chunk boundaries (64 ms) — reduces clicks and discontinuities
CROSSFADE_SAMPLES = 1024


def _cosine_crossfade(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Blend two 1D tensors with cosine crossfade. len(a) == len(b)."""
    n = a.shape[0]
    t = torch.linspace(0, 1, n + 2, device=a.device, dtype=a.dtype)[1:-1]
    fade_out = 0.5 * (1 + torch.cos(t * math.pi))   # 1 -> 0
    fade_in = 0.5 * (1 + torch.cos((1 - t) * math.pi))  # 0 -> 1
    return fade_out * a + fade_in * b


def match_chunked(
    knn_vc,
    query_seq: torch.Tensor,
    matching_set: torch.Tensor,
    topk: int = 4,
    chunk_frames: int | None = None,
    progress_cb=None,
) -> torch.Tensor:
    """Run knn_vc.match() in chunks to avoid CUDA OOM on long audio.

    Uses overlap and cosine crossfade at chunk boundaries to avoid clicks and
    discontinuities from the vocoder. kNN lookup is O(N·M); HiFiGAN is the
    memory bottleneck.
    """
    if chunk_frames is None:
        chunk_frames = MATCH_CHUNK_FRAMES
    device = str(next(knn_vc.parameters()).device)
    total = query_seq.shape[0]
    if total <= chunk_frames:
        msg = f"Vocoding {total} frames..."
        print(msg)
        if progress_cb:
            progress_cb(msg)
        return knn_vc.match(query_seq, matching_set, topk=topk, device=device)

    n_chunks = math.ceil(total / chunk_frames)
    crossfade = min(CROSSFADE_SAMPLES, (MATCH_OVERLAP_FRAMES * _WAVLM_HOP) // 2)
    parts: list[torch.Tensor] = []
    right_tail_prev: torch.Tensor | None = None

    for i in range(n_chunks):
        start = i * chunk_frames
        end = min(start + chunk_frames, total)
        s_ext = max(0, start - MATCH_OVERLAP_FRAMES)
        e_ext = min(total, end + MATCH_OVERLAP_FRAMES)

        wav = knn_vc.match(query_seq[s_ext:e_ext], matching_set, topk=topk, device=device).squeeze()
        trim_start = (start - s_ext) * _WAVLM_HOP
        trim_end = (e_ext - end) * _WAVLM_HOP
        L = wav.shape[0]

        if trim_end < crossfade or (L - trim_start - trim_end) < crossfade:
            wav = wav[trim_start : L - trim_end if trim_end else L]
            parts.append(wav)
            right_tail_prev = None
        else:
            left_tail = wav[trim_start : trim_start + crossfade]
            main = wav[trim_start + crossfade : L - trim_end - crossfade]
            right_tail = wav[L - trim_end - crossfade : L - trim_end]

            if right_tail_prev is not None:
                blended = _cosine_crossfade(right_tail_prev.to(wav.device), left_tail)
                parts.append(blended)
            else:
                parts.append(left_tail)

            parts.append(main)
            right_tail_prev = right_tail

        msg = f"Vocoding: chunk {i+1}/{n_chunks} ({end}/{total} frames)"
        print(msg)
        if progress_cb:
            progress_cb(msg)
        torch.cuda.empty_cache()

    if right_tail_prev is not None:
        parts.append(right_tail_prev)

    return torch.cat(parts).unsqueeze(0)


@contextlib.contextmanager
def log_progress(label: str, interval: float = 5.0, progress_cb=None):
    """Context manager: prints elapsed time every `interval` seconds while a block runs.

    Usage:
        with log_progress("kNN matching 12345 frames", progress_cb=log):
            out_wav = knn_vc.match(...)
    """
    start = time.time()
    stop = threading.Event()

    def _ticker():
        while not stop.wait(interval):
            elapsed = time.time() - start
            msg = f"{label} ... {elapsed:.0f}s"
            print(msg)
            if progress_cb:
                progress_cb(msg)

    t = threading.Thread(target=_ticker, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join(timeout=interval + 1)
        elapsed = time.time() - start
        msg = f"{label} done ({elapsed:.1f}s)"
        print(msg)
        if progress_cb:
            progress_cb(msg)
