"""
OpenVoice-based voice conversion (tone color transfer).

Uses openvoice-cli (OpenVoice stage 2) for natural voice-to-voice conversion
with a single reference sample. Supports chunked processing to reduce VRAM
and CPU fallback on OOM.

Requires: pip install openvoice-cli
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

try:
    from openvoice_cli.__main__ import tune_one
    import openvoice_cli.se_extractor as _se_extractor
    from openvoice_cli.api import ToneColorConverter
    from openvoice_cli.downloader import download_checkpoint
    import openvoice_cli.__main__ as _ov_main
except ImportError:
    tune_one = None
    _se_extractor = None
    ToneColorConverter = None
    download_checkpoint = None
    _ov_main = None


def _patch_openvoice_short_audio() -> None:
    """Allow short audio in openvoice_cli: use at least 1 segment instead of asserting 'input audio is too short'."""
    if _se_extractor is None:
        return
    _orig = getattr(_se_extractor, "split_audio_vad", None)
    if not _orig or getattr(_se_extractor.split_audio_vad, "_short_audio_patched", False):
        return
    import numpy as np

    def _split_vad_allow_short(audio_path, audio_name, target_dir, split_seconds=10.0):
        from pydub import AudioSegment
        from whisper_timestamped.transcribe import get_audio_tensor, get_vad_segments
        import os
        SAMPLE_RATE = 16000
        audio = AudioSegment.from_file(audio_path)
        audio_vad = get_audio_tensor(audio_path)
        segments = get_vad_segments(
            audio_vad, output_sample=True, min_speech_duration=0.1,
            min_silence_duration=1, method="silero",
        )
        segments = [(float(seg["start"]) / SAMPLE_RATE, float(seg["end"]) / SAMPLE_RATE) for seg in segments]
        audio_active = AudioSegment.silent(duration=0)
        for st, et in segments:
            audio_active += audio[int(st * 1000) : int(et * 1000)]
        audio_dur = audio_active.duration_seconds
        if audio_dur < 0.5:
            audio_active = audio
            audio_dur = audio_active.duration_seconds
        wavs_folder = os.path.join(target_dir, audio_name, "wavs")
        os.makedirs(wavs_folder, exist_ok=True)
        num_splits = max(1, int(np.round(audio_dur / split_seconds)))
        interval = audio_dur / num_splits
        start_time = 0.0
        for count in range(num_splits):
            end_time = audio_dur if count == num_splits - 1 else min(start_time + interval, audio_dur)
            seg = audio_active[int(start_time * 1000) : int(end_time * 1000)]
            seg.export(os.path.join(wavs_folder, f"{audio_name}_seg{count}.wav"), format="wav")
            start_time = end_time
        return wavs_folder

    _split_vad_allow_short._short_audio_patched = True
    _se_extractor.split_audio_vad = _split_vad_allow_short


# Chunk duration in seconds — smaller = less VRAM, more chunks
DEFAULT_CHUNK_DURATION_SEC = 18
# Minimum chunk length (seconds) — avoid empty/tiny chunks that break resampler
MIN_CHUNK_DURATION_SEC = 1.5
# Crossfade at chunk boundaries (samples at 22.05 kHz typical for OpenVoice)
CROSSFADE_SAMPLES = 2205  # ~0.1 s


def _get_converter_and_ref(ref_path: str, device: str, report_fn=None):
    """Load ToneColorConverter once and extract reference embedding. Reuse for all chunks."""
    import os
    current_dir = os.path.dirname(_ov_main.__file__)
    ckpt_converter = os.path.join(current_dir, "checkpoints", "converter")
    if not os.path.exists(ckpt_converter):
        os.makedirs(ckpt_converter, exist_ok=True)
        download_checkpoint(ckpt_converter)
    if report_fn:
        report_fn("Loading OpenVoice model (once)...", None)
    converter = ToneColorConverter(
        os.path.join(ckpt_converter, "config.json"),
        device=device,
    )
    converter.load_ckpt(os.path.join(ckpt_converter, "checkpoint.pth"))
    if report_fn:
        report_fn("Extracting reference voice embedding...", None)
    target_se, _ = _se_extractor.get_se(ref_path, converter, vad=True)
    return converter, target_se


def _log(msg: str) -> None:
    print(f"[OpenVoice] {msg}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()


def is_available() -> bool:
    return tune_one is not None


def _cosine_crossfade(a: list, b: list, fade_samples: int) -> list:
    """Blend end of a with start of b over fade_samples. Returns blended segment (length fade_samples)."""
    import numpy as np
    tail = a[-fade_samples:]
    head = b[:fade_samples]
    n = len(tail)
    t = np.linspace(0, 1, n, dtype=np.float32)
    fade_out = 0.5 * (1 + np.cos(t * np.pi))
    fade_in = 0.5 * (1 + np.cos((1 - t) * np.pi))
    return (np.array(tail, dtype=np.float64) * fade_out + np.array(head, dtype=np.float64) * fade_in).tolist()


def convert(
    input_path: str,
    ref_path: str,
    output_path: str | None = None,
    device: str | None = None,
    *,
    tau: float = 0.65,
    temperature: float = 1.0,
    chunk_duration_sec: float | None = DEFAULT_CHUNK_DURATION_SEC,
    progress_cb: callable | None = None,
    use_cpu_fallback_on_oom: bool = True,
) -> str:
    """
    Convert voice in input audio to match the reference speaker (tone color transfer).
    tau: strength of reference voice (0.3–1.0). Higher = more like reference.
    temperature: voice expressiveness (0.7–1.3). Higher = warmer/more variation, lower = calmer.
    Effective tau is clamped to [0.2, 1.0]. If chunk_duration_sec is set, processes in chunks.
    """
    if not is_available():
        raise RuntimeError(
            "OpenVoice is not installed. Install with: pip install openvoice-cli"
        )

    _patch_openvoice_short_audio()

    import librosa as _librosa
    import numpy as np
    import soundfile as sf

    effective_tau = max(0.2, min(1.0, float(tau) * float(temperature)))

    def report(msg: str, p: float | None = None) -> None:
        _log(msg)
        if progress_cb:
            progress_cb(msg, p)

    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    _log(f"Device: {device}")

    input_path = str(Path(input_path).resolve())
    ref_path = str(Path(ref_path).resolve())

    # Load full input to get duration and split into chunks
    y_in, sr_in = _librosa.load(input_path, sr=None, mono=True)
    duration_sec = len(y_in) / sr_in
    report(f"Input: {duration_sec:.1f} s, {sr_in} Hz")

    if output_path is None:
        fd = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        output_path = fd.name
        fd.close()
    output_path = str(Path(output_path).resolve())

    chunk_sec = chunk_duration_sec if chunk_duration_sec and chunk_duration_sec > 0 else None
    if chunk_sec is None or duration_sec <= chunk_sec:
        # Single pass with tau
        report("Converting in one pass...", 0.2)
        try:
            converter, target_se = _get_converter_and_ref(ref_path, device, report)
            source_se, _ = _se_extractor.get_se(input_path, converter, vad=True)
            converter.convert(
                audio_src_path=input_path,
                src_se=source_se,
                tgt_se=target_se,
                output_path=output_path,
                tau=effective_tau,
            )
        except RuntimeError as e:
            if use_cpu_fallback_on_oom and "out of memory" in str(e).lower() and device != "cpu":
                _log("GPU OOM — retrying on CPU...")
                report("Converting on CPU (slower)...", 0.3)
                tune_one(input_file=input_path, ref_file=ref_path, output_file=output_path, device="cpu")
            else:
                raise
        report("Done.", 1.0)
        return output_path

    # Chunked: load model and reference once, then convert each chunk with get_se + convert
    chunk_samples = int(chunk_sec * sr_in)
    min_chunk_samples = max(1, int(MIN_CHUNK_DURATION_SEC * sr_in))
    # Build ranges so no chunk is shorter than min (merge short tail into previous)
    ranges = []
    pos = 0
    while pos < len(y_in):
        end = min(pos + chunk_samples, len(y_in))
        if ranges and (end - pos) < min_chunk_samples:
            ranges[-1] = (ranges[-1][0], end)
        else:
            ranges.append((pos, end))
        pos = end
    n_chunks = len(ranges)
    report(f"Chunked conversion: {n_chunks} chunks of ~{chunk_sec:.0f} s (model loaded once)", 0.05)

    try:
        converter, target_se = _get_converter_and_ref(ref_path, device, report)
    except Exception as e:
        raise RuntimeError(f"Failed to load OpenVoice model or reference: {e}") from e

    out_sr = None
    segments = []

    for i, (start, end) in enumerate(ranges):
        chunk_wav = y_in[start:end]
        report(f"Chunk {i + 1}/{n_chunks} ({start / sr_in:.1f}–{end / sr_in:.1f} s)", 0.1 + 0.75 * (i / n_chunks))

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_in:
            sf.write(f_in.name, chunk_wav, sr_in)
            chunk_in_path = f_in.name
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_out:
            chunk_out_path = f_out.name

        def _run_chunk(dev: str) -> None:
            source_se, _ = _se_extractor.get_se(chunk_in_path, converter, vad=True)
            converter.convert(
                audio_src_path=chunk_in_path,
                src_se=source_se,
                tgt_se=target_se,
                output_path=chunk_out_path,
                tau=effective_tau,
            )

        try:
            _run_chunk(device)
        except RuntimeError as e:
            if use_cpu_fallback_on_oom and "out of memory" in str(e).lower() and device != "cpu":
                _log("GPU OOM in chunk — retrying chunk on CPU (one-off)...")
                tune_one(input_file=chunk_in_path, ref_file=ref_path, output_file=chunk_out_path, device="cpu")
            else:
                Path(chunk_in_path).unlink(missing_ok=True)
                raise
        Path(chunk_in_path).unlink(missing_ok=True)

        out_chunk, out_sr = _librosa.load(chunk_out_path, sr=None, mono=True)
        Path(chunk_out_path).unlink(missing_ok=True)
        if out_sr is None:
            out_sr = sr_in  # fallback
        segments.append(out_chunk.tolist())

    # Concatenate with crossfade between chunks
    report("Concatenating chunks...", 0.9)
    fade = min(CROSSFADE_SAMPLES, len(segments[0]) // 2) if segments else 0
    if fade > 0 and len(segments) > 1:
        result = list(segments[0][:-fade])
        for i in range(1, len(segments)):
            blended = _cosine_crossfade(segments[i - 1], segments[i], fade)
            result.extend(blended)
            rest = segments[i][fade:-fade] if i < len(segments) - 1 else segments[i][fade:]
            result.extend(rest)
        out_array = np.array(result, dtype=np.float32)
    else:
        out_array = np.concatenate([np.array(s, dtype=np.float32) for s in segments], axis=0)

    sf.write(output_path, out_array, out_sr or sr_in)
    report("Done.", 1.0)
    return output_path
