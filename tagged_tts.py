#!/usr/bin/env python3
"""
Tagged TTS pipeline (text-only, no timestamps).

Goal:
  - User pastes a script containing tags like: [śmiech], [westchnienie], [pauza]
  - Speech is synthesized using a voice-cloning TTS (XTTS / Chatterbox / F5 / Edge fallback)
  - Non-speech tag events are synthesized using Bark (optional) so the laughter/sigh
    is present in the generated WAV (no subscription, local if installed).

This module is UI-agnostic; app.py provides the Gradio tab.
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000

# Script tags: everything inside [ ... ]
_TAG_RE = re.compile(r"(\[[^\[\]]+\])")


def _silence(seconds: float) -> np.ndarray:
    n = max(0, int(seconds * SAMPLE_RATE))
    return np.zeros(n, dtype=np.float32)


def _normalize_text(text: str) -> str:
    # Preserve punctuation (helps prosody). Normalize whitespace a bit.
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def split_script(text: str) -> list[tuple[str, str]]:
    """
    Returns list of ("speech"|"tag", chunk).
    Keeps tags as separate items.
    """
    t = _normalize_text(text)
    if not t:
        return []
    parts = _TAG_RE.split(t)
    out: list[tuple[str, str]] = []
    for p in parts:
        if not p:
            continue
        if p.startswith("[") and p.endswith("]"):
            tag = p[1:-1].strip()
            if tag:
                out.append(("tag", tag))
        else:
            s = p.strip()
            if s:
                out.append(("speech", s))
    return out


_PL_TO_BARK = {
    "śmiech": "[laughter]",
    "delikatny śmiech": "[laughs]",
    "śmiech krótki": "[laughs]",
    "chichot": "[laughs]",
    "westchnienie": "[sighs]",
    "westchnienie głębokie": "[sighs]",
    "odchrząknięcie": "[clears throat]",
    "wdech": "[gasps]",
    "pauza": "...",
    "pauza długa": "... ...",
}


def bark_token_for_tag(tag: str) -> str | None:
    """Map Polish tag label to Bark token/prompt snippet."""
    key = re.sub(r"\s+", " ", tag.strip().lower())
    return _PL_TO_BARK.get(key)


def _resample_to_16k(wav: np.ndarray, sr: int) -> np.ndarray:
    if sr == SAMPLE_RATE:
        return wav.astype(np.float32, copy=False)
    return librosa.resample(wav.astype(np.float32, copy=False), orig_sr=sr, target_sr=SAMPLE_RATE)


def synth_speech_chunk(
    text: str,
    engine: str,
    language: str,
    ref_audio_path: str | None,
    *,
    xtts_speed: float = 1.0,
    chatterbox_exaggeration: float = 0.5,
    chatterbox_cfg_weight: float = 0.5,
    f5tts_ref_text: str = "",
    f5tts_model_path: str | None = None,
    f5tts_speed: float = 1.0,
    edge_voice: str = "pl-PL-ZofiaNeural",
    edge_rate: str = "+0%",
    edge_pitch: str = "+0Hz",
) -> np.ndarray:
    """
    Synthesize speech (no tags) to 16kHz mono float32.
    engine: "chatterbox" | "xtts" | "f5tts" | "edge"
    """
    engine = (engine or "").lower().strip()
    txt = text.strip()
    if not txt:
        return np.zeros(0, dtype=np.float32)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out_path = tmp.name

    try:
        if engine == "xtts":
            import xtts_engine
            if not ref_audio_path or not Path(ref_audio_path).exists():
                raise RuntimeError("XTTS requires a reference voice sample.")
            if not xtts_engine.is_available():
                raise RuntimeError("XTTS not installed (coqui-tts missing).")
            xtts_engine.synthesize(
                text=txt,
                language=language,
                ref_audio_path=ref_audio_path,
                output_path=out_path,
                speed=float(xtts_speed),
            )

        elif engine == "chatterbox":
            import chatterbox_engine
            if not ref_audio_path or not Path(ref_audio_path).exists():
                raise RuntimeError("Chatterbox requires a reference voice sample.")
            if not chatterbox_engine.is_available():
                raise RuntimeError("Chatterbox TTS not available (chatterbox-tts missing).")
            chatterbox_engine.synthesize(
                text=txt,
                language=language,
                ref_audio_path=ref_audio_path,
                output_path=out_path,
                exaggeration=float(chatterbox_exaggeration),
                cfg_weight=float(chatterbox_cfg_weight),
            )

        elif engine == "f5tts":
            import f5tts_engine
            if not ref_audio_path or not Path(ref_audio_path).exists():
                raise RuntimeError("F5-TTS requires a reference voice sample.")
            if not f5tts_engine.is_available():
                raise RuntimeError("F5-TTS not installed (f5-tts missing).")
            f5tts_engine.synthesize(
                text=txt,
                ref_audio_path=ref_audio_path,
                output_path=out_path,
                ref_text=f5tts_ref_text or "",
                model_path=f5tts_model_path if f5tts_model_path else None,
                speed=float(f5tts_speed),
            )

        else:
            # Edge TTS (online)
            import edge_tts

            # edge_tts outputs mp3 via Communicate.save, then we load+resample.
            Path(out_path).unlink(missing_ok=True)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as mp3:
                mp3_path = mp3.name
            async def _run():
                comm = edge_tts.Communicate(txt, voice=edge_voice, rate=edge_rate, pitch=edge_pitch)
                await comm.save(mp3_path)
            import asyncio
            asyncio.run(_run())
            wav, sr = librosa.load(mp3_path, sr=None, mono=True)
            Path(mp3_path).unlink(missing_ok=True)
            wav16 = _resample_to_16k(wav, int(sr))
            sf.write(out_path, wav16, SAMPLE_RATE)

        wav, sr = librosa.load(out_path, sr=None, mono=True)
        return _resample_to_16k(wav, int(sr))
    finally:
        Path(out_path).unlink(missing_ok=True)


def synth_tag_bark(tag: str, language: str, bark_preset: str) -> np.ndarray:
    """
    Synthesize a non-speech event (laughter/sigh) via Bark if installed.
    Returns 16kHz mono.
    """
    token = bark_token_for_tag(tag)
    if token is None:
        return np.zeros(0, dtype=np.float32)
    try:
        from bark import SAMPLE_RATE as bark_sr, generate_audio, preload_models
    except Exception:
        # Bark is optional — if missing, caller should fall back (e.g. pause/ignore).
        raise RuntimeError("Bark not installed. Install: pip install git+https://github.com/suno-ai/bark.git")

    preload_models()
    preset = bark_preset.strip() or "v2/pl_speaker_0"
    audio = generate_audio(token, history_prompt=preset)
    wav = np.asarray(audio, dtype=np.float32).squeeze()
    if wav.ndim > 1:
        wav = wav.mean(axis=0)
    return _resample_to_16k(wav, int(bark_sr))


def render_tagged_script(
    script: str,
    *,
    speech_engine: str,
    language: str,
    ref_audio_path: str | None,
    use_bark_for_tags: bool,
    bark_preset: str,
    pause_between_chunks_s: float = 0.15,
) -> np.ndarray:
    """
    Main entry: returns final 16kHz wav.
    """
    chunks = split_script(script)
    if not chunks:
        raise ValueError("Empty script.")

    out: list[np.ndarray] = []
    for kind, payload in chunks:
        if kind == "speech":
            wav = synth_speech_chunk(
                payload,
                engine=speech_engine,
                language=language,
                ref_audio_path=ref_audio_path,
            )
            out.append(wav)
            out.append(_silence(pause_between_chunks_s))
        else:
            # tag
            key = payload.strip().lower()
            if key in ("pauza", "pause"):
                out.append(_silence(0.35))
                continue
            if key in ("pauza długa", "long pause"):
                out.append(_silence(0.8))
                continue
            if use_bark_for_tags and bark_token_for_tag(payload) is not None:
                try:
                    out.append(synth_tag_bark(payload, language=language, bark_preset=bark_preset))
                    out.append(_silence(0.1))
                except RuntimeError:
                    # Bark missing or failed — degrade gracefully instead of failing whole render
                    out.append(_silence(0.25))
            # unknown tags are simply ignored in audio

    wav_out = np.concatenate([w.astype(np.float32, copy=False) for w in out if w.size > 0]) if out else np.zeros(0, dtype=np.float32)
    # Soft peak normalize
    peak = float(np.max(np.abs(wav_out)) + 1e-9) if wav_out.size else 0.0
    if peak > 1.0:
        wav_out = wav_out / peak
    return wav_out.astype(np.float32, copy=False)

