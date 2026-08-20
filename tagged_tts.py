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

from audio_utils import OUTPUT_SR_CLONE, OUTPUT_SR_EDGE, concat_with_crossfade, normalize_lufs, output_sr_for_backend, resample
from text_normalize_pl import normalize_for_tts

# Legacy alias for app.py
SAMPLE_RATE = OUTPUT_SR_EDGE

# Script tags: everything inside [ ... ]
_TAG_RE = re.compile(r"(\[[^\[\]]+\])")


def _silence(seconds: float, sr: int) -> np.ndarray:
    n = max(0, int(seconds * sr))
    return np.zeros(n, dtype=np.float32)


def _normalize_text(text: str) -> str:
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


def _target_sr_for_engine(engine: str) -> int:
    return output_sr_for_backend(engine if engine != "edge" else "edge")


def synth_speech_chunk(
    text: str,
    engine: str,
    language: str,
    ref_audio_path: str | None,
    *,
    xtts_speed: float = 1.0,
    chatterbox_exaggeration: float = 0.35,
    chatterbox_cfg_weight: float = 0.7,
    f5tts_ref_text: str = "",
    f5tts_model_path: str | None = None,
    f5tts_speed: float = 1.0,
    edge_voice: str = "pl-PL-ZofiaNeural",
    edge_rate: str = "+0%",
    edge_pitch: str = "+0Hz",
) -> tuple[np.ndarray, int]:
    """
    Synthesize speech (no tags). Returns (wav, sample_rate).
    engine: "chatterbox" | "xtts" | "f5tts" | "edge"
    """
    engine = (engine or "").lower().strip()
    target_sr = _target_sr_for_engine(engine)
    txt = normalize_for_tts(text.strip(), language)
    if not txt:
        return np.zeros(0, dtype=np.float32), target_sr

    sub_chunks: list[str]
    if engine == "chatterbox":
        import chatterbox_engine
        sub_chunks = chatterbox_engine.split_text_for_tts(txt)
    else:
        sub_chunks = [txt]

    part_wavs: list[np.ndarray] = []

    for sub in sub_chunks:
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
                    text=sub,
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
                m = chatterbox_engine.get_model()
                if not getattr(m, "_is_multilingual", False) and (language or "en").strip().lower() != "en":
                    from translate import edge_voice_for_lang_code
                    return synth_speech_chunk(
                        sub,
                        engine="edge",
                        language=language,
                        ref_audio_path=ref_audio_path,
                        edge_voice=edge_voice_for_lang_code(language),
                        edge_rate=edge_rate,
                        edge_pitch=edge_pitch,
                        xtts_speed=xtts_speed,
                        chatterbox_exaggeration=chatterbox_exaggeration,
                        chatterbox_cfg_weight=chatterbox_cfg_weight,
                        f5tts_ref_text=f5tts_ref_text,
                        f5tts_model_path=f5tts_model_path,
                        f5tts_speed=f5tts_speed,
                    )
                chatterbox_engine.synthesize(
                    text=sub,
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
                model_path = f5tts_model_path
                if not model_path and (language or "").strip().lower() == "pl":
                    model_path = "polish"
                f5tts_engine.synthesize(
                    text=sub,
                    ref_audio_path=ref_audio_path,
                    output_path=out_path,
                    ref_text=f5tts_ref_text or "",
                    model_path=model_path,
                    speed=float(f5tts_speed),
                )

            else:
                import edge_tts
                Path(out_path).unlink(missing_ok=True)
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as mp3:
                    mp3_path = mp3.name
                async def _run():
                    comm = edge_tts.Communicate(sub, voice=edge_voice, rate=edge_rate, pitch=edge_pitch)
                    await comm.save(mp3_path)
                import asyncio
                asyncio.run(_run())
                wav, sr = librosa.load(mp3_path, sr=None, mono=True)
                Path(mp3_path).unlink(missing_ok=True)
                wav = resample(wav, int(sr), target_sr)
                sf.write(out_path, wav, target_sr)

            wav, sr = librosa.load(out_path, sr=None, mono=True)
            part_wavs.append(resample(wav, int(sr), target_sr))
        finally:
            Path(out_path).unlink(missing_ok=True)

    if not part_wavs:
        return np.zeros(0, dtype=np.float32), target_sr
    if len(part_wavs) == 1:
        return part_wavs[0], target_sr
    return concat_with_crossfade(part_wavs, target_sr, pause_s=0.05, fade_ms=40), target_sr


def synth_tag_bark(tag: str, language: str, bark_preset: str, target_sr: int) -> np.ndarray:
    """
    Synthesize a non-speech event (laughter/sigh) via Bark if installed.
    """
    token = bark_token_for_tag(tag)
    if token is None:
        return np.zeros(0, dtype=np.float32)
    try:
        from bark import SAMPLE_RATE as bark_sr, generate_audio, preload_models
    except Exception:
        raise RuntimeError("Bark not installed. Install: pip install git+https://github.com/suno-ai/bark.git")

    preload_models()
    preset = bark_preset.strip() or "v2/pl_speaker_0"
    audio = generate_audio(token, history_prompt=preset)
    wav = np.asarray(audio, dtype=np.float32).squeeze()
    if wav.ndim > 1:
        wav = wav.mean(axis=0)
    return resample(wav, int(bark_sr), target_sr)


def render_tagged_script(
    script: str,
    *,
    speech_engine: str,
    language: str,
    ref_audio_path: str | None,
    use_bark_for_tags: bool,
    bark_preset: str,
    pause_between_chunks_s: float = 0.05,
    xtts_speed: float = 1.0,
    chatterbox_exaggeration: float = 0.35,
    chatterbox_cfg_weight: float = 0.7,
    f5tts_ref_text: str = "",
    f5tts_model_path: str | None = None,
    f5tts_speed: float = 1.0,
) -> tuple[np.ndarray, int]:
    """
    Main entry: returns (final wav, sample_rate).
    """
    chunks = split_script(script)
    if not chunks:
        raise ValueError("Empty script.")

    engine = (speech_engine or "edge").lower().strip()
    if not f5tts_model_path and (language or "").strip().lower() == "pl" and engine == "f5tts":
        f5tts_model_path = "polish"

    sr = _target_sr_for_engine(engine)
    out_parts: list[np.ndarray] = []

    for kind, payload in chunks:
        if kind == "speech":
            wav, chunk_sr = synth_speech_chunk(
                payload,
                engine=engine,
                language=language,
                ref_audio_path=ref_audio_path,
                xtts_speed=xtts_speed,
                chatterbox_exaggeration=chatterbox_exaggeration,
                chatterbox_cfg_weight=chatterbox_cfg_weight,
                f5tts_ref_text=f5tts_ref_text,
                f5tts_model_path=f5tts_model_path,
                f5tts_speed=f5tts_speed,
            )
            sr = chunk_sr
            if wav.size:
                out_parts.append(wav)
                out_parts.append(_silence(pause_between_chunks_s, sr))
        else:
            key = payload.strip().lower()
            if key in ("pauza", "pause"):
                out_parts.append(_silence(0.35, sr))
                continue
            if key in ("pauza długa", "long pause"):
                out_parts.append(_silence(0.8, sr))
                continue
            if use_bark_for_tags and bark_token_for_tag(payload) is not None:
                try:
                    out_parts.append(synth_tag_bark(payload, language=language, bark_preset=bark_preset, target_sr=sr))
                    out_parts.append(_silence(0.1, sr))
                except RuntimeError:
                    out_parts.append(_silence(0.25, sr))

    if not out_parts:
        return np.zeros(0, dtype=np.float32), sr

    wav_out = np.concatenate([p.astype(np.float32, copy=False) for p in out_parts if p.size])
    wav_out = normalize_lufs(wav_out, sr)
    return wav_out, sr
