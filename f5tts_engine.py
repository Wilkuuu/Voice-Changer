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

# Gregniuki repo — NOTE: Polish/*.pt checkpoints ship with a pinyin vocab (not Polish).
# Do NOT use for Polish TTS; kept only for manual/experimental overrides.
_POLISH_REPO = "Gregniuki/F5-tts_English_German_Polish"
_POLISH_CKPT = "Polish/model_500000.pt"
_POLISH_VOCAB = "Polish/vocab.txt"
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

    vocab_file = ""
    cache_key = model_path

    # Resolve "polish" shorthand — community checkpoint uses pinyin vocab (broken for PL).
    if model_path == "polish":
        print(
            "[F5-TTS] WARNING: Gregniuki 'polish' checkpoint uses a pinyin vocab and produces "
            "garbled output for Polish text. Falling back to F5TTS_v1_Base (zero-shot). "
            "For Polish, use Chatterbox Multilingual instead.",
            flush=True,
        )
        model_path = None
        cache_key = "base"

    if model_path == "polish_legacy":
        from huggingface_hub import hf_hub_download
        ckpt = hf_hub_download(repo_id=_POLISH_REPO, filename=_POLISH_CKPT)
        vocab_file = hf_hub_download(repo_id=_POLISH_REPO, filename=_POLISH_VOCAB)
        model_path = ckpt
        cache_key = "polish_legacy"
        print(f"[F5-TTS] Legacy Polish checkpoint (experimental): {ckpt}", flush=True)

    if _tts is None or cache_key != _loaded_model_path:
        from f5_tts.api import F5TTS
        if model_path:
            print(f"[F5-TTS] Loading custom model: {model_path}")
            kwargs = {"ckpt_file": model_path}
            if vocab_file:
                kwargs["vocab_file"] = vocab_file
            _tts = F5TTS(**kwargs)
        else:
            print("[F5-TTS] Loading base F5TTS_v1_Base model...")
            _tts = F5TTS(model_type="F5TTS_v1_Base")
        _loaded_model_path = cache_key
        print("[F5-TTS] Model loaded.")
    return _tts


def _ref_text_cache_path(ref_audio_path: str) -> Path:
    p = Path(ref_audio_path)
    st = p.stat()
    key = hashlib.sha1(f"{p.resolve()}:{st.st_mtime}:{st.st_size}".encode()).hexdigest()[:16]
    return p.with_suffix(f".reftext{key}.txt")


def get_or_transcribe_ref_text(
    ref_audio_path: str,
    ref_text: str = "",
    language: str = "pl",
) -> str:
    """
    Return reference transcript for F5-TTS. Uses explicit ``ref_text``, disk cache,
    or Whisper auto-transcription on the reference file.
    """
    explicit = (ref_text or "").strip()
    if explicit:
        return explicit

    lang = (language or "pl").strip().lower()
    cache_key = f"{Path(ref_audio_path).resolve()}:{lang}"
    if cache_key in _ref_text_cache:
        return _ref_text_cache[cache_key]

    cache_path = _ref_text_cache_path(ref_audio_path)
    # Include language in sidecar to avoid reusing wrong-language transcripts.
    lang_cache = cache_path.with_name(cache_path.stem + f".{lang}.txt")
    if lang_cache.exists():
        text = lang_cache.read_text(encoding="utf-8").strip()
        if text:
            _ref_text_cache[cache_key] = text
            return text

    try:
        from translate import get_whisper
        model = get_whisper(model_size="base")
        segments, info = model.transcribe(
            ref_audio_path,
            beam_size=5,
            vad_filter=True,
            language=lang if lang != "auto" else None,
            task="transcribe",
        )
        text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())
        detected = getattr(info, "language", lang)
        print(f"[F5-TTS] Auto ref_text (lang={detected}, {len(text)} chars): {text[:80]}...", flush=True)
    except Exception as e:
        print(f"[F5-TTS] Auto ref_text transcription failed ({e}); using empty ref_text.", flush=True)
        text = ""

    if text:
        try:
            lang_cache.write_text(text, encoding="utf-8")
        except Exception:
            pass
        _ref_text_cache[cache_key] = text
    return text


def synthesize(
    text: str,
    ref_audio_path: str,
    output_path: str,
    ref_text: str = "",
    model_path: str | None = None,
    speed: float = 1.0,
    seed: int = -1,
    language: str = "pl",
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

    resolved_ref_text = get_or_transcribe_ref_text(ref_audio_path, ref_text, language=language)
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
