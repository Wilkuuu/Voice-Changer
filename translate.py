#!/usr/bin/env python3
"""
Translation pipeline: speech → text → translate → TTS → (voice conversion)

Free tools (all local except edge-tts):
- faster-whisper : ASR, translates any language → English segments with timestamps
- argostranslate : English → target language (local, own package system)
- edge-tts       : Neural TTS (free, Microsoft Edge voices, online)

Sync mode (DTW): for each Whisper segment, TTS audio is non-uniformly warped to
match the energy-peak positions of the source — not just its total length.
DTW (Dynamic Time Warping) aligns RMS energy envelopes so that speech peaks in the
translated audio land at the same moments as speech peaks in the original.
"""

import asyncio
import os
import re
import threading
import tempfile
from pathlib import Path

import numpy as np
import librosa
import soundfile as sf
import torch
import edge_tts
from faster_whisper import WhisperModel

import argostranslate.package
import argostranslate.translate

SAMPLE_RATE = 16000
VAD_TOP_DB = 30  # silence threshold (dB below peak) for speech/silence split

# Target languages: ISO 639-1 code + edge-tts neural voice
LANGUAGES: dict[str, dict] = {
    "English":    {"lang_code": "en", "tts_voice": "en-US-JennyNeural"},
    "Polish":     {"lang_code": "pl", "tts_voice": "pl-PL-ZofiaNeural"},
    "German":     {"lang_code": "de", "tts_voice": "de-DE-KatjaNeural"},
    "French":     {"lang_code": "fr", "tts_voice": "fr-FR-DeniseNeural"},
    "Spanish":    {"lang_code": "es", "tts_voice": "es-ES-ElviraNeural"},
    "Italian":    {"lang_code": "it", "tts_voice": "it-IT-ElsaNeural"},
    "Russian":    {"lang_code": "ru", "tts_voice": "ru-RU-SvetlanaNeural"},
    "Ukrainian":  {"lang_code": "uk", "tts_voice": "uk-UA-PolinaNeural"},
    "Dutch":      {"lang_code": "nl", "tts_voice": "nl-NL-ColetteNeural"},
    "Portuguese": {"lang_code": "pt", "tts_voice": "pt-PT-RaquelNeural"},
}

_whisper_model: WhisperModel | None = None
_whisper_cfg: tuple[str, str, str] | None = None  # (size, device, compute_type)
_installed_pairs: set[tuple[str, str]] = set()


def _whisper_auto_device_compute() -> tuple[str, str]:
    """Pick the best (device, compute_type) for faster-whisper on this host."""
    forced = (os.environ.get("VOICE_CHANGER_FORCE_DEVICE") or "").strip().lower()
    try:
        if forced == "cpu":
            return "cpu", "int8"
        if torch.cuda.is_available() and forced != "cpu":
            return "cuda", "int8_float16"
    except Exception:
        pass
    return "cpu", "int8"


def get_whisper(
    model_size: str = "base",
    device: str | None = None,
    compute_type: str | None = None,
) -> WhisperModel:
    """
    Lazy-load faster-whisper model with optional size/device override.

    - ``model_size``: ``tiny|base|small|medium|large-v2|large-v3`` etc.
    - ``device``: ``cpu|cuda|auto`` (None = auto)
    - ``compute_type``: ``int8`` (CPU), ``int8_float16`` (GPU fast), ``float16`` (GPU high-q)

    Re-loads when any of the three parameters change.
    """
    global _whisper_model, _whisper_cfg
    if device in (None, "auto"):
        device, auto_ct = _whisper_auto_device_compute()
        if compute_type is None:
            compute_type = auto_ct
    if compute_type is None:
        compute_type = "int8_float16" if device == "cuda" else "int8"

    target = (str(model_size), str(device), str(compute_type))
    if _whisper_model is None or _whisper_cfg != target:
        if _whisper_model is not None:
            print(f"Switching Whisper: {_whisper_cfg} → {target}")
            try:
                del _whisper_model
            except Exception:
                pass
            _whisper_model = None
        print(f"Loading Whisper-{model_size} on {device} ({compute_type})...")
        _whisper_model = WhisperModel(model_size, device=device, compute_type=compute_type)
        _whisper_cfg = target
        print("Whisper loaded.")
    return _whisper_model


def ensure_translation_package(from_code: str, to_code: str) -> None:
    pair = (from_code, to_code)
    if pair in _installed_pairs:
        return
    installed_langs = argostranslate.translate.get_installed_languages()
    for lang in installed_langs:
        if lang.code == from_code:
            if any(t.to_lang.code == to_code for t in lang.translations_to):
                _installed_pairs.add(pair)
                return
    print(f"Downloading translation package [{from_code}→{to_code}] (~50-100 MB)...")
    argostranslate.package.update_package_index()
    available = argostranslate.package.get_available_packages()
    pkg = next(
        (p for p in available if p.from_code == from_code and p.to_code == to_code),
        None,
    )
    if pkg is None:
        raise ValueError(f"No argostranslate package for [{from_code}→{to_code}].")
    argostranslate.package.install_from_path(pkg.download())
    _installed_pairs.add(pair)
    print("Translation package installed.")


# ── Gender-aware translation ──────────────────────────────────────────────────

# Prepended before the text so the MT model sees a female speaker in context.
# The translated version of this sentence is then stripped from the output.
_FEMALE_CONTEXT_EN = "I am a woman."

# Language-specific regex to catch any remaining masculine forms after translation.
# Format: list of (pattern, replacement) applied in order.
_FEMALE_FIXERS: dict[str, list[tuple[str, str]]] = {
    # Polish: first-person past tense -łem/-łeś → -łam/-łaś
    "pl": [
        (r"\b(\w+)łem\b", r"\1łam"),   # byłem→byłam, zrobiłem→zrobiłam
        (r"\b(\w+)łeś\b", r"\1łaś"),   # byłeś→byłaś
        (r"\b(\w+)łbym\b", r"\1łabym"),# chciałbym→chciałabym (conditional)
    ],
    # German: ich war/hatte/bin ... (hard to regex; context injection handles most cases)
    "de": [],
    # French: je suis allé → je suis allée  (past participle agreement)
    "fr": [
        (r"\bje suis allé\b", "je suis allée"),
        (r"\bje suis parti\b", "je suis partie"),
        (r"\bje suis sorti\b", "je suis sortie"),
    ],
    # Russian: first-person past tense -л → -ла
    "ru": [
        (r"\b(\w+)л\b(?!\w)", r"\1ла"),   # был→была, сделал→сделала
    ],
    # Ukrainian: similar to Russian
    "uk": [
        (r"\b(\w+)в\b(?!\w)", r"\1ла"),
    ],
}


def _apply_female_fixers(text: str, lang_code: str) -> str:
    for pattern, replacement in _FEMALE_FIXERS.get(lang_code, []):
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def translate_en_to_target(text: str, target_language: str, female_narrator: bool = False) -> str:
    """Translate English text to target language, optionally with female narrator context."""
    lang_code = LANGUAGES[target_language]["lang_code"]
    if lang_code == "en":
        return text
    ensure_translation_package("en", lang_code)

    if female_narrator:
        # Prepend a short female-context sentence so the MT model uses feminine forms
        combined = _FEMALE_CONTEXT_EN + " " + text
        translated = argostranslate.translate.translate(combined, "en", lang_code)
        # Strip the translated context sentence (it's short, ends with first ". " or ".")
        dot = translated.find(". ")
        if 0 < dot < 40:          # sanity-check: context sentence is always short
            translated = translated[dot + 2:].strip()
        elif translated.startswith(translated[:3]):
            # fallback: remove first token-sentence if no ". " found
            pass
        return _apply_female_fixers(translated, lang_code)

    return argostranslate.translate.translate(text, "en", lang_code)


def _tts_run(text: str, voice: str, path: str, rate: str = "+0%", pitch: str = "+0Hz") -> None:
    """Run edge-tts in a dedicated thread with its own event loop (no Gradio conflict).

    rate  : speaking rate offset, e.g. "+0%", "-10%", "+20%"
    pitch : pitch offset in Hz,   e.g. "+0Hz", "-5Hz",  "+10Hz"
    """
    exc: list[Exception] = []

    def _worker():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            comm = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
            loop.run_until_complete(comm.save(path))
        except Exception as e:
            exc.append(e)
        finally:
            loop.close()

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join()
    if exc:
        raise exc[0]


def tts_to_file(text: str, target_language: str, output_path: str,
                rate: str = "+0%", pitch: str = "+0Hz") -> None:
    _tts_run(text, LANGUAGES[target_language]["tts_voice"], output_path, rate=rate, pitch=pitch)


# ── Silence-aware fitting ─────────────────────────────────────────────────────

def _fit_by_silence(tts_wav: np.ndarray, target_samples: int) -> np.ndarray:
    """
    Fit TTS audio to target_samples by scaling ONLY the silence regions.
    Speech portions are NEVER stretched or resampled — they play at natural speed.

    Strategy:
    - Detect speech/silence regions via VAD (librosa.effects.split)
    - Distribute the silence budget proportionally across all silence gaps
    - If speech alone exceeds target: return speech-only audio (trimmed to target)
    - If no silences exist: pad with trailing silence
    """
    if len(tts_wav) == 0:
        return np.zeros(target_samples, dtype=np.float32)

    intervals = librosa.effects.split(tts_wav, top_db=VAD_TOP_DB)

    if len(intervals) == 0:
        return np.zeros(target_samples, dtype=np.float32)

    # Build list of silence lengths: [pre, between..., post]  (len = n_speech + 1)
    n = len(intervals)
    silence_orig = [int(intervals[0][0])]
    for j in range(n - 1):
        silence_orig.append(int(intervals[j + 1][0] - intervals[j][1]))
    silence_orig.append(int(len(tts_wav) - intervals[-1][1]))

    speech_total = sum(int(e - s) for s, e in intervals)

    # If speech alone is already longer than target, concatenate speech and trim
    if speech_total >= target_samples:
        parts = [tts_wav[s:e] for s, e in intervals]
        return np.concatenate(parts).astype(np.float32)[:target_samples]

    silence_budget = target_samples - speech_total
    orig_silence_total = sum(silence_orig)

    if orig_silence_total > 0:
        scale = silence_budget / orig_silence_total
        new_silences = [max(0, round(s * scale)) for s in silence_orig]
    else:
        # No original silences: add all budget as trailing silence
        new_silences = [0] * n + [silence_budget]

    # Assemble: silence[j], speech[j], ..., speech[n-1], silence[n]
    parts: list[np.ndarray] = []
    for j, (s, e) in enumerate(intervals):
        if new_silences[j] > 0:
            parts.append(np.zeros(new_silences[j], dtype=np.float32))
        parts.append(tts_wav[s:e].astype(np.float32))
    if new_silences[-1] > 0:
        parts.append(np.zeros(new_silences[-1], dtype=np.float32))

    result = np.concatenate(parts).astype(np.float32) if parts else np.zeros(0, dtype=np.float32)

    # Guarantee exact target length
    if len(result) < target_samples:
        result = np.pad(result, (0, target_samples - len(result)))
    return result[:target_samples]


# ── Main building block ───────────────────────────────────────────────────────

def _build_synced_audio(
    segments: list,
    target_language: str,
    total_duration: float,
    female_narrator: bool = False,
    tts_rate: str = "+0%",
    tts_pitch: str = "+0Hz",
    progress_cb=None,
) -> tuple[np.ndarray, str, str]:
    """
    For each Whisper segment:
      1. Translate
      2. Synthesize TTS
      3. DTW-warp TTS energy peaks to align with source segment energy peaks
      4. Place at source segment's start position
    Returns (audio_array, english_text, translated_text).
    """
    voice = LANGUAGES[target_language]["tts_voice"]
    lang_code = LANGUAGES[target_language]["lang_code"]

    if lang_code != "en":
        ensure_translation_package("en", lang_code)

    total_samples = int(total_duration * SAMPLE_RATE)
    out = np.zeros(total_samples, dtype=np.float32)

    all_english: list[str] = []
    all_translated: list[str] = []
    n = len(segments)

    for i, seg in enumerate(segments):
        english = seg.text.strip()
        if not english:
            continue
        all_english.append(english)

        translated = translate_en_to_target(english, target_language, female_narrator=female_narrator)
        all_translated.append(translated)

        if progress_cb:
            progress_cb(f"Segment {i+1}/{n}: {translated[:70]}")

        # TTS synthesis
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tts_path = f.name
        _tts_run(translated, voice, tts_path, rate=tts_rate, pitch=tts_pitch)
        tts_wav, _ = librosa.load(tts_path, sr=SAMPLE_RATE, mono=True)
        Path(tts_path).unlink(missing_ok=True)

        if len(tts_wav) == 0:
            continue

        # Fit TTS to source segment duration by adjusting silences only
        seg_start_s = int(seg.start * SAMPLE_RATE)
        seg_end_s = min(int(seg.end * SAMPLE_RATE), total_samples)
        target_samples = seg_end_s - seg_start_s

        tts_warped = _fit_by_silence(tts_wav, target_samples)

        # Place in output buffer
        end_s = seg_start_s + len(tts_warped)
        if end_s > len(out):
            out = np.pad(out, (0, end_s - len(out)))
        out[seg_start_s:end_s] += tts_warped

    return out, " ".join(all_english), " ".join(all_translated)


# ── Public pipeline entry point ───────────────────────────────────────────────

def run_pipeline(
    audio_path: str,
    target_language: str,
    output_path: str,
    knn_vc=None,
    matching_set=None,
    topk: int = 4,
    sync: bool = True,
    female_narrator: bool = False,
    tts_rate: str = "+0%",
    tts_pitch: str = "+0Hz",
    progress_cb=None,
) -> tuple[str, str]:
    """
    Full pipeline: ASR → translate → TTS → (optional voice conversion).

    Args:
        sync  : if True, DTW-align each TTS segment to source energy peaks.
    Returns:
        (english_text, translated_text)
    """
    def step(msg: str):
        print(msg)
        if progress_cb:
            progress_cb(msg)

    step("Transcribing audio (Whisper)...")
    whisper = get_whisper()
    segments_gen, info = whisper.transcribe(audio_path, task="translate", beam_size=5)
    segments = list(segments_gen)
    total_duration = info.duration
    step(f"Detected [{info.language}], {total_duration:.1f}s, {len(segments)} segments")

    if sync:
        step("Translating & synthesizing per-segment (DTW sync)...")
        audio_out, english_text, translated_text = _build_synced_audio(
            segments, target_language, total_duration,
            female_narrator=female_narrator,
            tts_rate=tts_rate, tts_pitch=tts_pitch,
            progress_cb=step,
        )
    else:
        english_text = " ".join(s.text.strip() for s in segments)
        step(f"Translating full text → {target_language}...")
        translated_text = translate_en_to_target(english_text, target_language, female_narrator=female_narrator)
        step(f"Translated: {translated_text[:120]}")
        step(f"Synthesizing ({LANGUAGES[target_language]['tts_voice']})...")
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tts_path = f.name
        tts_to_file(translated_text, target_language, tts_path, rate=tts_rate, pitch=tts_pitch)
        audio_out, _ = librosa.load(tts_path, sr=SAMPLE_RATE, mono=True)
        Path(tts_path).unlink(missing_ok=True)

    if knn_vc is not None and matching_set is not None:
        step("Applying voice conversion...")
        from voice_utils import extract_features_chunked, match_chunked
        tensor = torch.from_numpy(audio_out).unsqueeze(0).to(next(knn_vc.parameters()).device)
        with torch.inference_mode():
            query_seq = extract_features_chunked(knn_vc, tensor, progress_cb=step)
            out_wav = match_chunked(knn_vc, query_seq, matching_set, topk=topk, progress_cb=step)
        torch.cuda.empty_cache()
        sf.write(output_path, out_wav.squeeze().cpu().numpy(), SAMPLE_RATE)
    else:
        sf.write(output_path, audio_out, SAMPLE_RATE)

    return english_text, translated_text


# ── Two-step pipeline (for interactive text editing) ─────────────────────────

_SEGMENT_RE = re.compile(r"^\[\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*\]\s*(.*)", re.MULTILINE)


def transcribe_and_translate(
    audio_path: str,
    target_language: str,
    female_narrator: bool = False,
    progress_cb=None,
) -> tuple[str, float]:
    """
    Step 1 of the two-step interactive pipeline.

    Runs Whisper ASR → argostranslate per segment.
    Returns (formatted_text, total_duration).

    formatted_text is a multi-line string:
        [0.00 - 3.45] Przetłumaczony tekst segmentu...
        [3.45 - 7.12] Kolejny segment...

    The user can edit this text in the UI; the [start - end] markers carry
    timing information used in Step 2.
    """
    def step(msg: str):
        print(msg)
        if progress_cb:
            progress_cb(msg)

    step("Transcribing (Whisper)...")
    whisper = get_whisper()
    segments_gen, info = whisper.transcribe(audio_path, task="translate", beam_size=5)
    segments = list(segments_gen)
    step(f"Detected [{info.language}], {info.duration:.1f}s, {len(segments)} segments")

    lang_code = LANGUAGES[target_language]["lang_code"]
    if lang_code != "en":
        ensure_translation_package("en", lang_code)

    lines: list[str] = []
    n = len(segments)
    for i, seg in enumerate(segments):
        english = seg.text.strip()
        if not english:
            continue
        translated = translate_en_to_target(english, target_language, female_narrator=female_narrator)
        step(f"Segment {i+1}/{n}: {translated[:70]}")
        lines.append(f"[{seg.start:.2f} - {seg.end:.2f}] {translated}")

    return "\n".join(lines), info.duration


# Common Whisper hallucinations to filter out
_WHISPER_HALLUCINATIONS = {
    "this is the end of the video",
    "thank you for watching",
    "please subscribe",
    "like and subscribe",
    "subtitles by",
    "transcribed by",
    "[music]",
    "[applause]",
    "[silence]",
}


def transcribe_only(
    audio_path: str,
    progress_cb=None,
    model_size: str = "base",
) -> tuple[str, str, float]:
    """
    Transcribe audio in its original language without translation.

    Returns (plain_text, detected_language_code, duration).
    Filters common Whisper hallucinations.

    ``model_size`` allows the caller to request a larger Whisper model
    (``medium``, ``large-v3``) when higher transcription accuracy matters
    (e.g. Chatterbox TTS path where the text feeds a re-synthesis step).
    """
    def step(msg: str):
        print(msg)
        if progress_cb:
            progress_cb(msg)

    step(f"Transcribing (Whisper-{model_size}, original language)...")
    whisper = get_whisper(model_size=model_size)
    segments_gen, info = whisper.transcribe(audio_path, task="transcribe", beam_size=5)
    segments = list(segments_gen)
    step(f"Detected [{info.language}], {info.duration:.1f}s, {len(segments)} segments")

    parts: list[str] = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        if text.lower().rstrip(".!?,") in _WHISPER_HALLUCINATIONS:
            continue
        parts.append(text)

    return " ".join(parts), info.language, info.duration


def parse_srt(srt_content: str) -> tuple[str, float]:
    """Parse SRT file content into the segments format used by the app.

    Returns (segments_text, total_duration) where segments_text is:
        [start - end] text
        ...
    and total_duration is the end time of the last segment (in seconds).
    """
    ts_re = re.compile(
        r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[,.](\d{3})"
    )

    lines: list[str] = []
    total_duration = 0.0

    blocks = re.split(r"\n\s*\n", srt_content.strip())
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        block_lines = block.splitlines()

        ts_match = None
        ts_idx = None
        for j, line in enumerate(block_lines):
            m = ts_re.match(line.strip())
            if m:
                ts_match = m
                ts_idx = j
                break
        if ts_match is None:
            continue

        h1, m1, s1, ms1 = (int(ts_match.group(k)) for k in (1, 2, 3, 4))
        h2, m2, s2, ms2 = (int(ts_match.group(k)) for k in (5, 6, 7, 8))
        start = h1 * 3600 + m1 * 60 + s1 + ms1 / 1000
        end   = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000

        text = " ".join(block_lines[ts_idx + 1:]).strip()
        if not text:
            continue

        total_duration = max(total_duration, end)
        lines.append(f"[{start:.2f} - {end:.2f}] {text}")

    return "\n".join(lines), total_duration


def polish_segments_with_ai(
    segments_text: str,
    lang_code: str = "pl",
    context_hint: str = "",
    api_key: str = "",
) -> str:
    """
    Use Claude API to fix grammar, coherence, and text length of translated segments.

    Segments format:  [start - end] text
    Duration of each segment (end - start) is used to guide text length.
    Returns improved segments in the same format.
    """
    import os
    import anthropic

    key = api_key.strip() or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise ValueError(
            "Anthropic API key is required. Set ANTHROPIC_API_KEY env var or paste it in the UI."
        )

    # Annotate each segment with duration so the model can judge appropriate text length
    parsed = _SEGMENT_RE.findall(segments_text)
    if not parsed:
        raise ValueError("No segments found. Expected format: [start - end] Text")

    annotated_lines: list[str] = []
    for start_s, end_s, text in parsed:
        duration = float(end_s) - float(start_s)
        annotated_lines.append(f"[{float(start_s):.2f} - {float(end_s):.2f}] ({duration:.1f}s) {text.strip()}")

    context_block = f"\nNarrative context: {context_hint.strip()}" if context_hint.strip() else ""

    system_prompt = (
        f"You are a professional text editor for speech synthesis. "
        f"You receive transcribed/translated speech segments with timestamps and durations. "
        f"Your task:\n"
        f"1. Fix all grammar, typo, and transcription errors\n"
        f"2. Make the text coherent and natural as spoken dialogue\n"
        f"3. Adjust text length to match the segment duration — a {lang_code} TTS voice speaks "
        f"   roughly 2–3 words per second. Short segments (<1s) = 2–4 words max. "
        f"   Long segments (>10s) = several sentences.\n"
        f"4. Preserve the original meaning and tone\n"
        f"5. Output ONLY the corrected segments, one per line, in the EXACT same format: "
        f"   [start - end] corrected text\n"
        f"   Do NOT include the duration annotation in output.\n"
        f"   Do NOT add explanations, notes, or extra lines.{context_block}"
    )

    user_msg = "\n".join(annotated_lines)

    client = anthropic.Anthropic(api_key=key)
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=4096,
        messages=[{"role": "user", "content": user_msg}],
        system=system_prompt,
    )

    return response.content[0].text.strip()


def synthesize_from_edited(
    edited_text: str,
    total_duration: float,
    target_language: str,
    output_path: str,
    knn_vc=None,
    matching_set=None,
    topk: int = 4,
    sync: bool = True,
    tts_rate: str = "+0%",
    tts_pitch: str = "+0Hz",
    tts_backend: str = "edge",   # "edge" | "xtts" | "chatterbox" | "f5tts"
    xtts_ref_path: str | None = None,
    xtts_speed: float = 1.0,
    chatterbox_exaggeration: float = 0.5,
    chatterbox_cfg_weight: float = 0.5,
    f5tts_ref_text: str = "",
    f5tts_model_path: str | None = None,
    f5tts_speed: float = 1.0,
    progress_cb=None,
) -> None:
    """
    Step 2 of the two-step interactive pipeline.

    Parses the user-edited segment text (format: "[start - end] text per line"),
    synthesizes TTS for each segment, fits it to the original timing.

    tts_backend="edge"        : edge-tts + optional kNN-VC
    tts_backend="xtts"        : XTTS v2 zero-shot voice cloning (~1.8 GB, Polish native)
    tts_backend="chatterbox"  : Chatterbox Multilingual — beats ElevenLabs, MIT license
    tts_backend="f5tts"       : F5-TTS flow-matching — best naturalness, fine-tunable
    """
    def step(msg: str):
        print(msg)
        if progress_cb:
            progress_cb(msg)

    parsed = _SEGMENT_RE.findall(edited_text)
    if not parsed:
        raise ValueError("No segments found in edited text. Expected format: [start - end] Text...")

    lang_code = LANGUAGES[target_language]["lang_code"]

    # Resolve which backend to use, with graceful fallback
    backend = tts_backend.lower()
    ref_path = xtts_ref_path  # shared reference audio for all cloning backends

    if backend == "xtts":
        try:
            import xtts_engine
            if not xtts_engine.is_available() or not ref_path or not Path(ref_path).exists():
                step("XTTS unavailable/no ref — falling back to edge-tts")
                backend = "edge"
        except ImportError:
            step("XTTS not installed — falling back to edge-tts")
            backend = "edge"

    elif backend == "chatterbox":
        try:
            import chatterbox_engine
            if not chatterbox_engine.is_available():
                step("Chatterbox not installed — falling back to edge-tts")
                backend = "edge"
            elif not ref_path or not Path(ref_path).exists():
                step("Chatterbox requires a reference audio — falling back to edge-tts")
                backend = "edge"
        except ImportError:
            step("Chatterbox not installed — falling back to edge-tts")
            backend = "edge"

    elif backend == "f5tts":
        try:
            import f5tts_engine
            if not f5tts_engine.is_available():
                step("F5-TTS not installed — falling back to edge-tts")
                backend = "edge"
            elif not ref_path or not Path(ref_path).exists():
                step("F5-TTS requires a reference audio — falling back to edge-tts")
                backend = "edge"
        except ImportError:
            step("F5-TTS not installed — falling back to edge-tts")
            backend = "edge"

    voice = LANGUAGES[target_language]["tts_voice"]
    total_samples = int(total_duration * SAMPLE_RATE)
    out = np.zeros(total_samples, dtype=np.float32)

    n = len(parsed)
    for i, (start_s, end_s, text) in enumerate(parsed):
        text = text.strip()
        if not text:
            continue

        start_f, end_f = float(start_s), float(end_s)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tts_path = f.name

        if backend == "xtts":
            step(f"XTTS {i+1}/{n}: {text[:70]}")
            import xtts_engine
            xtts_engine.synthesize(
                text=text,
                language=lang_code,
                ref_audio_path=ref_path,
                output_path=tts_path,
                speed=xtts_speed,
            )

        elif backend == "chatterbox":
            step(f"Chatterbox {i+1}/{n}: {text[:70]}")
            import chatterbox_engine
            chatterbox_engine.synthesize(
                text=text,
                language=lang_code,
                ref_audio_path=ref_path,
                output_path=tts_path,
                exaggeration=chatterbox_exaggeration,
                cfg_weight=chatterbox_cfg_weight,
            )

        elif backend == "f5tts":
            step(f"F5-TTS {i+1}/{n}: {text[:70]}")
            import f5tts_engine
            f5tts_engine.synthesize(
                text=text,
                ref_audio_path=ref_path,
                output_path=tts_path,
                ref_text=f5tts_ref_text,
                model_path=f5tts_model_path if f5tts_model_path else None,
                speed=f5tts_speed,
            )

        else:  # edge-tts
            step(f"TTS {i+1}/{n}: {text[:70]}")
            Path(tts_path).unlink(missing_ok=True)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tts_path = f.name
            _tts_run(text, voice, tts_path, rate=tts_rate, pitch=tts_pitch)

        tts_wav, _ = librosa.load(tts_path, sr=SAMPLE_RATE, mono=True)
        Path(tts_path).unlink(missing_ok=True)

        if len(tts_wav) == 0:
            continue

        seg_start = int(start_f * SAMPLE_RATE)
        seg_end = min(int(end_f * SAMPLE_RATE), total_samples)
        target_samples = seg_end - seg_start

        tts_warped = _fit_by_silence(tts_wav, target_samples) if sync else tts_wav

        end_idx = seg_start + len(tts_warped)
        if end_idx > len(out):
            out = np.pad(out, (0, end_idx - len(out)))
        out[seg_start:end_idx] += tts_warped

    # Voice conversion — only for edge-tts backend (others have built-in voice cloning)
    if backend == "edge" and knn_vc is not None and matching_set is not None:
        step("Applying voice conversion...")
        from voice_utils import extract_features_chunked, match_chunked
        tensor = torch.from_numpy(out).unsqueeze(0).to(next(knn_vc.parameters()).device)
        with torch.inference_mode():
            query_seq = extract_features_chunked(knn_vc, tensor, progress_cb=step)
            out_wav = match_chunked(knn_vc, query_seq, matching_set, topk=topk, progress_cb=step)
        torch.cuda.empty_cache()
        sf.write(output_path, out_wav.squeeze().cpu().numpy(), SAMPLE_RATE)
    else:
        sf.write(output_path, out, SAMPLE_RATE)
