"""
F5-TTS engine — flow-matching TTS with zero-shot voice cloning.

Highest naturalness scores in 2025 benchmarks (RTF ~0.15, DiT architecture).
Supports Polish via community fine-tune checkpoint or zero-shot cloning.

Install:
    pip install f5-tts

Community Polish checkpoint (optional, better quality for Polish):
    huggingface.co/Gregniuki/F5-tts_English_German_Polish
    Trained on ~90 hours of Polish audio (A100, 24h training).

Fine-tuning your own voice:
    Run: python finetune_f5tts.py --help
    Minimum: 10 hours of clean single-speaker audio for good quality.
    Start from the Polish community checkpoint rather than the base model.

Note: F5-TTS requires a text transcript of the reference audio for best results.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_tts = None
_loaded_model_path: str | None = None
_ref_text_cache: dict[str, str] = {}


def is_available() -> bool:
    try:
        from f5_tts.api import F5TTS  # noqa: F401
        return True
    except ImportError:
        return False


def get_tts(model_path: str | None = None):
    """
    Load F5-TTS model. Caches the model; reloads if model_path changes.

    Args:
        model_path : path to a custom/fine-tuned .pt checkpoint, or None for base model.
                     Set to "polish" to auto-download the community Polish checkpoint.
    """
    global _tts, _loaded_model_path

    # Resolve "polish" shorthand to community checkpoint
    if model_path == "polish":
        from huggingface_hub import hf_hub_download
        model_path = hf_hub_download(
            repo_id="Gregniuki/F5-tts_English_German_Polish",
            filename="model_1096000.pt",
        )
        print(f"[F5-TTS] Polish community checkpoint: {model_path}")

    if _tts is None or model_path != _loaded_model_path:
        from f5_tts.api import F5TTS
        if model_path:
            print(f"[F5-TTS] Loading custom model: {model_path}")
            _tts = F5TTS(ckpt_file=model_path)
        else:
            print("[F5-TTS] Loading base F5TTS_v1_Base model...")
            _tts = F5TTS(model_type="F5TTS_v1_Base")
        _loaded_model_path = model_path
        print("[F5-TTS] Model loaded.")
    return _tts


def _ref_text_cache_path(ref_audio_path: str) -> Path:
    p = Path(ref_audio_path)
    st = p.stat()
    key = hashlib.sha1(f"{p.resolve()}:{st.st_mtime}:{st.st_size}".encode()).hexdigest()[:16]
    return p.with_suffix(f".reftext{key}.txt")


def get_or_transcribe_ref_text(ref_audio_path: str, ref_text: str = "") -> str:
    """
    Return reference transcript for F5-TTS. Uses explicit ``ref_text``, disk cache,
    or Whisper auto-transcription on the reference file.
    """
    explicit = (ref_text or "").strip()
    if explicit:
        return explicit

    cache_key = str(Path(ref_audio_path).resolve())
    if cache_key in _ref_text_cache:
        return _ref_text_cache[cache_key]

    cache_path = _ref_text_cache_path(ref_audio_path)
    if cache_path.exists():
        text = cache_path.read_text(encoding="utf-8").strip()
        if text:
            _ref_text_cache[cache_key] = text
            return text

    try:
        from translate import get_whisper
        model = get_whisper(model_size="base")
        segments, _info = model.transcribe(ref_audio_path, beam_size=3, vad_filter=True)
        text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())
    except Exception as e:
        print(f"[F5-TTS] Auto ref_text transcription failed ({e}); using empty ref_text.", flush=True)
        text = ""

    if text:
        try:
            cache_path.write_text(text, encoding="utf-8")
        except Exception:
            pass
        _ref_text_cache[cache_key] = text
        print(f"[F5-TTS] Auto ref_text ({len(text)} chars): {text[:80]}...", flush=True)
    return text


def synthesize(
    text: str,
    ref_audio_path: str,
    output_path: str,
    ref_text: str = "",
    model_path: str | None = None,
    speed: float = 1.0,
    seed: int = -1,
) -> None:
    """
    Synthesize text with zero-shot voice cloning via flow matching.

    Args:
        text           : text to synthesize
        ref_audio_path : reference speaker audio — 5–15 s recommended
        output_path    : output .wav path
        ref_text       : transcript of reference audio (improves accuracy; leave empty
                         to auto-transcribe with Whisper)
        model_path     : path to custom .pt checkpoint, "polish" for the Polish
                         community model, or None for the base English model
        speed          : speaking speed multiplier (0.5–2.0, default 1.0)
        seed           : random seed for reproducibility (-1 = random)
    """
    import soundfile as sf

    resolved_ref_text = get_or_transcribe_ref_text(ref_audio_path, ref_text)
    tts = get_tts(model_path)
    wav, sr, _ = tts.infer(
        ref_file=ref_audio_path,
        ref_text=resolved_ref_text,
        gen_text=text,
        speed=speed,
        seed=None if seed < 0 else seed,
    )
    sf.write(output_path, wav, sr)


def unload() -> None:
    """Free GPU memory by unloading the F5-TTS model."""
    global _tts
    if _tts is not None:
        try:
            import torch
            del _tts
            _tts = None
            torch.cuda.empty_cache()
            print("[F5-TTS] Model unloaded.")
        except Exception:
            _tts = None
