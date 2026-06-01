"""
Chatterbox Multilingual TTS engine — zero-shot voice cloning, native Polish support.

Beats ElevenLabs in blind preference tests (63.75% user preference).
Supports 23 languages natively including Polish — no fine-tuning needed.

Install:
    pip install chatterbox-tts

Model: ResembleAI/chatterbox (MIT license)
VRAM: ~6–8 GB (Multilingual), ~4 GB (English-only fallback)

Note: Every output embeds Resemble AI's PerTh neural watermark (inaudible).
"""

from __future__ import annotations

import re

from device_utils import empty_cache as _empty_cache, select_device as _select_device

# Chatterbox degrades / cuts off on long single-shot text (~200–350 chars). Chunk by sentence.
MAX_CHARS_PER_GENERATE = 220
PAUSE_BETWEEN_CHUNKS_S = 0.08

_model = None
_device: str | None = None


def _patch_alignment_stream_analyzer() -> None:
    """
    Chatterbox crashes on very short text (e.g. "Co?") when the alignment matrix
    has fewer than ~5 columns: A[completed_at:, :-5].max(dim=1) raises IndexError.
    """
    try:
        from chatterbox.models.t3.inference.alignment_stream_analyzer import (
            AlignmentStreamAnalyzer,
        )

        if getattr(AlignmentStreamAnalyzer, "_idx_patch", False):
            return

        import logging

        import torch

        logger = logging.getLogger(__name__)
        _orig_step = AlignmentStreamAnalyzer.step

        def _patched_step(self, logits, next_token=None):
            try:
                return _orig_step(self, logits, next_token=next_token)
            except IndexError:
                logger.warning(
                    "[Chatterbox] alignment_stream_analyzer IndexError (short text); forcing EOS"
                )
                logits = -(2**15) * torch.ones_like(logits)
                logits[..., self.eos_idx] = 2**15
                self.curr_frame_pos += 1
                return logits

        AlignmentStreamAnalyzer.step = _patched_step
        AlignmentStreamAnalyzer._idx_patch = True
    except Exception:
        pass


def _ensure_min_tts_length(text: str, min_len: int = 12) -> str:
    """Avoid alignment_stream_analyzer edge cases on ultra-short strings."""
    t = (text or "").strip()
    if len(t) >= min_len:
        return t
    # Ellipsis often adds tokens without changing meaning much for dubbing.
    return t + "…" * max(1, min_len - len(t))


def _patch_transformers_sdpa() -> None:
    """
    Patch transformers 5.x compatibility issue with Chatterbox.

    Chatterbox's AlignmentStreamAnalyzer sets output_attentions=True on a
    transformer config, but transformers 5.x raises ValueError when the model
    uses attn_implementation='sdpa'. Fix: switch to 'eager' before setting it.
    """
    try:
        from chatterbox.models.t3.inference.alignment_stream_analyzer import (
            AlignmentStreamAnalyzer,
        )

        if getattr(AlignmentStreamAnalyzer, "_sdpa_patched", False):
            return

        _orig_spy = AlignmentStreamAnalyzer._add_attention_spy

        def _patched_spy(self, tfmr, i, layer_idx, head_idx):
            cfg = getattr(tfmr, "config", None)
            if cfg is not None and getattr(cfg, "_attn_implementation", None) == "sdpa":
                cfg._attn_implementation = "eager"
            _orig_spy(self, tfmr, i, layer_idx, head_idx)

        AlignmentStreamAnalyzer._add_attention_spy = _patched_spy
        AlignmentStreamAnalyzer._sdpa_patched = True
    except Exception:
        pass  # patch is best-effort; if it fails the original error will surface

SUPPORTED_LANGUAGES = {
    "pl", "en", "de", "fr", "es", "it", "ru", "pt", "nl",
    "tr", "ar", "zh", "ja", "ko", "uk", "cs", "hu",
    "sv", "da", "fi", "no", "ro",
}


def is_available() -> bool:
    try:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS  # noqa: F401
        return True
    except ImportError:
        try:
            from chatterbox.tts import ChatterboxTTS  # noqa: F401
            return True
        except ImportError:
            return False


def get_model():
    global _model, _device
    if _model is None:
        _patch_transformers_sdpa()
        _patch_alignment_stream_analyzer()
        _device = _select_device()
        try:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            print(f"[Chatterbox] Loading Multilingual model on {_device}...")
            _model = ChatterboxMultilingualTTS.from_pretrained(device=_device)
            _model._is_multilingual = True
            print("[Chatterbox] Multilingual model loaded.")
        except (ImportError, Exception) as e:
            print(f"[Chatterbox] Multilingual unavailable ({e}), falling back to English-only model.")
            from chatterbox.tts import ChatterboxTTS
            _model = ChatterboxTTS.from_pretrained(device=_device)
            _model._is_multilingual = False
            print("[Chatterbox] English-only model loaded.")
    return _model


def split_text_for_tts(text: str, max_chars: int = MAX_CHARS_PER_GENERATE) -> list[str]:
    """Split text into sentence-sized chunks (Chatterbox cuts off on long single calls)."""
    t = re.sub(r"\s+", " ", (text or "").strip())
    if not t:
        return []
    if len(t) <= max_chars:
        return [t]

    sentences = re.split(r"(?<=[.!?…])\s+", t)
    chunks: list[str] = []
    current = ""

    def _flush() -> None:
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(sent) > max_chars:
            _flush()
            words = sent.split()
            buf = ""
            for w in words:
                if len(buf) + len(w) + 1 <= max_chars:
                    buf = f"{buf} {w}".strip()
                else:
                    if buf:
                        chunks.append(buf)
                    buf = w
            if buf:
                chunks.append(buf)
            continue
        if not current:
            current = sent
        elif len(current) + 1 + len(sent) <= max_chars:
            current = f"{current} {sent}"
        else:
            _flush()
            current = sent
    _flush()
    return chunks if chunks else [t[:max_chars]]


def synthesize(
    text: str,
    language: str,
    ref_audio_path: str,
    output_path: str,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
) -> None:
    """
    Synthesize text with zero-shot voice cloning.

    Args:
        text           : text to synthesize (in target language)
        language       : ISO 639-1 code ("pl", "en", ...) — used for Multilingual model
        ref_audio_path : reference speaker audio — 10–30 s recommended
        output_path    : output .wav path
        exaggeration   : emotion/expressiveness intensity (0.0–1.0, default 0.5)
        cfg_weight     : classifier-free guidance weight (0.0–1.0, default 0.5)
    """
    import numpy as np
    import soundfile as sf

    model = get_model()
    lang = language if language in SUPPORTED_LANGUAGES else "pl"
    chunks = split_text_for_tts(text)

    kwargs = dict(
        audio_prompt_path=ref_audio_path,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
    )
    if getattr(model, "_is_multilingual", False):
        kwargs["language_id"] = lang

    parts: list[np.ndarray] = []
    pause_n = max(0, int(PAUSE_BETWEEN_CHUNKS_S * model.sr))
    pause = np.zeros(pause_n, dtype=np.float32) if pause_n else None

    for i, chunk in enumerate(chunks):
        chunk = _ensure_min_tts_length(chunk)
        if len(chunks) > 1:
            print(f"[Chatterbox] TTS chunk {i + 1}/{len(chunks)} ({len(chunk)} chars)", flush=True)
        try:
            wav = model.generate(chunk, **kwargs)
        except IndexError:
            padded = _ensure_min_tts_length(chunk, min_len=20)
            if padded == chunk:
                raise
            print(f"[Chatterbox] Retrying after IndexError: {padded!r}", flush=True)
            wav = model.generate(padded, **kwargs)
        arr = wav.detach().cpu().numpy().astype(np.float32)
        if arr.ndim == 2:
            arr = arr.mean(axis=0)
        if arr.size == 0:
            print(f"[Chatterbox] Warning: empty audio for chunk {i + 1}, skipping", flush=True)
            continue
        parts.append(arr)
        if pause is not None and i < len(chunks) - 1:
            parts.append(pause)

    if not parts:
        raise RuntimeError("Chatterbox produced no audio (all chunks empty or failed).")

    out = np.concatenate(parts).astype(np.float32, copy=False)
    peak = float(np.abs(out).max())
    if peak > 1e-6:
        out = out / peak * 0.95
    sf.write(output_path, out, int(model.sr))


def unload() -> None:
    """Free GPU memory by unloading the Chatterbox TTS model."""
    global _model
    if _model is not None:
        try:
            del _model
            _model = None
            _empty_cache()
            print("[Chatterbox] TTS model unloaded.")
        except Exception:
            _model = None


# ── Voice Conversion (Chatterbox VC) ─────────────────────────────────────────

_vc_model = None


def is_vc_available() -> bool:
    try:
        from chatterbox.vc import ChatterboxVC  # noqa: F401
        return True
    except ImportError:
        return False


def get_vc_model():
    global _vc_model, _device
    if _vc_model is None:
        if _device is None:
            _device = _select_device()
        print(f"[Chatterbox VC] Loading voice conversion model on {_device}...")
        from chatterbox.vc import ChatterboxVC
        _vc_model = ChatterboxVC.from_pretrained(device=_device)
        print("[Chatterbox VC] Model loaded.")
    return _vc_model


def convert_voice(
    input_path: str,
    ref_path: str,
    output_path: str | None = None,
    seed: int | None = None,
) -> tuple[str | None, "np.ndarray", int]:  # type: ignore[name-defined]
    """
    Convert the voice in input_path to match the speaker in ref_path.

    Returns a tuple ``(output_path_or_None, wav_float32, sample_rate)``. Keeps
    the model's native sample rate (24 kHz for Chatterbox VC) — callers that
    need 16 kHz should resample downstream.

    Args:
        input_path  : source audio (content to preserve)
        ref_path    : reference speaker audio (voice identity to apply)
        output_path : optional .wav path; if None the file is not written
        seed        : optional seed (for best-of-N selection)
    """
    import numpy as np
    import torch
    import torchaudio as ta

    model = get_vc_model()

    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    wav = model.generate(audio=input_path, target_voice_path=ref_path)
    sr = int(getattr(model, "sr", 24000))

    if output_path is not None:
        ta.save(output_path, wav, sr)

    arr = wav.detach().cpu().numpy().astype(np.float32)
    if arr.ndim == 2:
        arr = arr.mean(axis=0)
    return output_path, arr, sr


def unload_vc() -> None:
    """Free GPU memory by unloading the Chatterbox VC model."""
    global _vc_model
    if _vc_model is not None:
        try:
            del _vc_model
            _vc_model = None
            _empty_cache()
            print("[Chatterbox VC] Model unloaded.")
        except Exception:
            _vc_model = None
