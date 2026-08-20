#!/usr/bin/env python3
"""
Gradio web interface for natural voice conversion (OpenVoice) and translation pipeline.

Voice Conversion tab: OpenVoice tone-color transfer (GPU, natural cloning).
Translate & Convert tab: Whisper + TTS + optional kNN-VC.

Usage:
    python app.py
    python app.py --share   # public URL via Gradio tunnel
    python app.py --port 7861
    python app.py --cpu --low-vram   # brak GPU / mało VRAM (wolniej, stabilniej)
"""

import sys

# Hide GPUs before torch/CUDA init — must run before `import torch`.
if "--cpu" in sys.argv:
    import os as _os_early

    _os_early.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import importlib
import importlib.util
import os
import tempfile
from pathlib import Path

# Must be set before CUDA initializes — reduces OOM from memory fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gradio as gr
import librosa
import torch

import translate as tr
import tagged_tts

try:
    from openvoice_engine import (
        DEFAULT_CHUNK_DURATION_SEC as _OV_DEFAULT_CHUNK_SEC,
        convert as openvoice_convert,
        is_available as openvoice_available,
    )
except ImportError:
    openvoice_available = lambda: False
    openvoice_convert = None
    _OV_DEFAULT_CHUNK_SEC = 18.0

SAMPLE_RATE = 16000
_model = None
# OpenVoice chunk length (seconds); main() may shrink in --low-vram mode.
_RUNTIME = {"openvoice_chunk_sec": float(_OV_DEFAULT_CHUNK_SEC)}


def _default_tts_backend(has_f5: bool, has_cb: bool) -> str:
    if has_f5:
        return "F5-TTS"
    if has_cb:
        return "Chatterbox"
    return "Edge-TTS + OpenVoice"


def _apply_openvoice_post_mix(
    wav_path: str,
    ref_path: str,
    output_path: str,
    tau: float = 0.4,
    progress_cb=None,
) -> None:
    """Light OpenVoice pass on full TTS mix for timbre consistency."""
    if not openvoice_available() or openvoice_convert is None:
        raise RuntimeError("OpenVoice post-step requested but openvoice-cli is not installed.")
    openvoice_convert(
        input_path=wav_path,
        ref_path=ref_path,
        output_path=output_path,
        chunk_duration_sec=_RUNTIME["openvoice_chunk_sec"],
        tau=tau,
        temperature=1.0,
        progress_cb=lambda msg, _p=None: progress_cb(msg) if progress_cb else None,
    )


def _has_bark() -> bool:
    try:
        import bark  # noqa: F401
        return True
    except Exception:
        return False


def _has_chatterbox_tts() -> bool:
    try:
        import chatterbox_engine
        return bool(chatterbox_engine.is_available())
    except Exception:
        return False


def _has_chatterbox_vc() -> bool:
    try:
        import chatterbox_engine
        return bool(chatterbox_engine.is_vc_available())
    except Exception:
        return False


def _has_xtts() -> bool:
    try:
        import xtts_engine
        return bool(xtts_engine.is_available())
    except Exception:
        return False


def _has_f5tts() -> bool:
    try:
        import f5tts_engine
        return bool(f5tts_engine.is_available())
    except Exception:
        return False


def _has_seedvc() -> bool:
    try:
        import seed_vc_engine
        return bool(seed_vc_engine.is_available())
    except Exception:
        return False


def _ensure_seedvc_engine_hot_reload() -> None:
    """
    Reload ``seed_vc_engine`` when its source file changed on disk.

    With ``docker compose`` bind-mounting ``.:/app``, ``git pull`` updates the
    ``.py`` but the old module object can stay in ``sys.modules`` until the
    process restarts — users then still hit the previous buggy loader. A
    mtime check fixes that without requiring ``docker compose restart`` every
    time.
    """
    spec = importlib.util.find_spec("seed_vc_engine")
    if spec is None or not getattr(spec, "origin", None):
        return
    disk_m = Path(spec.origin).stat().st_mtime
    sm = sys.modules.get("seed_vc_engine")
    if sm is None:
        return
    rec = getattr(sm, "_ENGINE_FILE_MTIME", None)
    if rec is None:
        return
    if abs(disk_m - rec) < 1e-9:
        return
    try:
        sm.unload()
    except Exception:
        pass
    importlib.reload(sm)
    print(
        f"[app] seed_vc_engine hot-reload (file mtime {rec:.6f} -> {disk_m:.6f})",
        flush=True,
    )


def _has_speaker_sim() -> bool:
    try:
        import speaker_sim
        return bool(speaker_sim.is_available())
    except Exception:
        return False


def get_model():
    global _model
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading kNN-VC model on {device}...")
        _model = torch.hub.load(
            "bshall/knn-vc",
            "knn_vc",
            prematched=True,
            trust_repo=True,
            device=device,
        )
        _model.eval()
        print("Model loaded.")
    return _model


def _audio_path(value):
    """Normalize Gradio Audio value to file path (str). Handles type='filepath' and dict from some Gradio 4 flows."""
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get("name") or value.get("path") or value.get("filename")
    return str(value) if value else None


def load_audio_array(path: str) -> torch.Tensor:
    try:
        wav, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, res_type="soxr_hq")
    except Exception:
        wav, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return torch.from_numpy(wav).unsqueeze(0)  # (1, T)


def _write_wav(arr, sr: int, suffix: str = ".wav") -> str:
    """Persist a float32 mono ndarray to a temp WAV file. Returns the path."""
    import soundfile as sf

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        out_path = tmp.name
    sf.write(out_path, arr, int(sr), subtype="PCM_16")
    return out_path


def _prepared_ref_path(ref_path: str, preprocess: bool, denoise: bool) -> tuple[str, str]:
    """
    Return (path_used_by_engine, status_text).

    When ``preprocess`` is True, runs ``ref_preprocess.prepare_reference`` and
    returns the sidecar cache path. Otherwise returns the original path.
    """
    if not preprocess:
        return ref_path, "reference: raw (preprocess off)"
    try:
        import ref_preprocess as rp

        _, _, meta = rp.prepare_reference(
            ref_path,
            target_sr=24_000,
            target_seconds=20.0,
            denoise=bool(denoise),
            use_cache=True,
        )
        status = (
            f"reference: {Path(meta.cached_path).name} "
            f"({meta.duration_sec:.1f}s, SNR~{meta.snr_db:.1f}dB"
            f"{', denoised' if meta.denoised else ''})"
        )
        return meta.cached_path, status
    except Exception as e:
        print(f"[run_conversion] preprocess failed ({e}) — using raw reference.", flush=True)
        return ref_path, f"reference: raw (preprocess failed: {e})"


def run_prepare_reference(
    source_audio,
    target_seconds: float,
    denoise: bool,
    use_cache: bool,
    progress=gr.Progress(),
):
    """
    Standalone tab: cut silence/noise, keep voiced speech, pick the cleanest window,
    normalize loudness — write a ready-to-use reference WAV (24 kHz mono).
    """
    src_path = _audio_path(source_audio)
    if not src_path or not Path(src_path).exists():
        raise gr.Error("Wgraj plik audio (WAV, MP3, FLAC, OGG…).")

    def log(msg: str, p: float | None = None) -> None:
        print(f"[Prepare Reference] {msg}", flush=True)
        if p is not None:
            progress(float(p), desc=msg)
        else:
            progress(None, desc=msg)

    log("Ładowanie audio…", 0.05)
    try:
        import ref_preprocess as rp
    except ImportError as e:
        raise gr.Error(
            "Brak modułu ref_preprocess. Upewnij się, że uruchamiasz app z katalogu projektu."
        ) from e

    sec = float(target_seconds or 20.0)
    sec = max(8.0, min(25.0, sec))

    log("VAD, opcjonalny denoise, wybór najlepszego fragmentu…", 0.2)
    try:
        _, sr, meta = rp.prepare_reference(
            src_path,
            target_sr=24_000,
            target_seconds=sec,
            denoise=bool(denoise),
            use_cache=bool(use_cache),
        )
    except Exception as e:
        raise gr.Error(f"Nie udało się przygotować referencji: {e}") from e

    out_path = Path(meta.cached_path)
    if not out_path.exists():
        raise gr.Error("Plik wyjściowy nie został zapisany — sprawdź uprawnienia do katalogu źródłowego.")

    log("Gotowe.", 1.0)

    src_dur_note = ""
    try:
        import librosa
        src_dur = float(librosa.get_duration(path=src_path))
        src_dur_note = f"- **Źródło (całość):** {src_dur:.1f} s\n"
    except Exception:
        pass

    cache_note = " (z cache)" if meta.used_cache else ""
    report = (
        f"### Przygotowana referencja{cache_note}\n\n"
        f"{src_dur_note}"
        f"- **Wyjście:** `{out_path.name}`\n"
        f"- **Długość:** {meta.duration_sec:.1f} s @ {meta.sr} Hz, mono\n"
        f"- **Po VAD (łącznie mowy):** {meta.voiced_seconds:.1f} s\n"
        f"- **Wybrane okno:** {meta.best_window_seconds:.1f} s\n"
        f"- **Szac. SNR:** {meta.snr_db:.1f} dB\n"
        f"- **Głośność:** {meta.loudness_lufs:.1f} LUFS\n"
        f"- **Denoise:** {'tak' if meta.denoised else 'nie'}\n"
        f"- **Pełna ścieżka:** `{out_path.resolve()}`\n\n"
        "Użyj tego pliku jako **Reference Voice** w Voice Conversion lub Translate. "
        "Zalecane 15–25 s czystej mowy jednego mówcy."
    )
    return str(out_path.resolve()), report


def _similarity_label(ref_wav_path: str, out_arr, out_sr: int) -> tuple[float, str]:
    try:
        import speaker_sim

        if not speaker_sim.is_available():
            return -1.0, "Speaker similarity: n/a (install speechbrain or resemblyzer)"
        score = speaker_sim.similarity(ref_wav_path, out_arr, out_sr=int(out_sr))
        if score < 0:
            return -1.0, "Speaker similarity: n/a"
        return score, f"Speaker similarity: {score:.2f} ({speaker_sim.label(score)})"
    except Exception as e:
        return -1.0, f"Speaker similarity: n/a ({e})"


def run_conversion(
    input_audio, reference_audio, vc_model,
    cb_exaggeration, cb_cfg_weight,
    tau, temperature,
    preprocess_ref, denoise_ref, best_of_n, apply_openvoice_post,
    whisper_size="base",
    progress=gr.Progress(),
):
    """
    Tab 1: Voice conversion. Supports Chatterbox VC, Seed-VC, OpenVoice,
    Chatterbox TTS, with preprocessing of the reference sample, best-of-N
    candidates, and optional OpenVoice post-processing.

    Returns
    -------
    (converted_audio_path, similarity_label)
    """
    input_path = _audio_path(input_audio)
    ref_path_raw = _audio_path(reference_audio)
    if not input_path or not Path(input_path).exists():
        raise gr.Error("Please upload an input audio file.")
    if not ref_path_raw or not Path(ref_path_raw).exists():
        raise gr.Error("Please upload a reference voice sample.")

    def log_progress(msg: str, p: float | None = None) -> None:
        print(f"[Voice Conversion] {msg}", flush=True)
        if p is not None:
            progress(float(p), desc=msg)
        else:
            progress(None, desc=msg)

    # Free VRAM held by *other* VC engines before loading the selected one.
    try:
        import device_utils

        keep_map = {
            "Chatterbox VC":  ("chatterbox_engine",),
            "Chatterbox TTS": ("chatterbox_engine",),
            "Seed-VC":        ("seed_vc_engine",),
            "OpenVoice":      ("openvoice_engine",),
        }
        device_utils.unload_all(except_modules=keep_map.get(vc_model, ()))
    except Exception as e:
        print(f"[run_conversion] unload_all failed: {e}", flush=True)

    ref_path, ref_status = _prepared_ref_path(
        ref_path_raw, preprocess=bool(preprocess_ref), denoise=bool(denoise_ref)
    )
    log_progress(ref_status, 0.05)

    n = max(1, int(best_of_n or 1))

    # ── Produce candidates ────────────────────────────────────────────────
    candidates: list[tuple[object, int]] = []  # (np.ndarray float32 mono, sr)
    engine_used = vc_model

    if vc_model == "Chatterbox VC":
        import chatterbox_engine
        if not chatterbox_engine.is_vc_available():
            raise gr.Error("Chatterbox VC not available. Run: pip install chatterbox-tts")
        log_progress("Loading Chatterbox VC model (first run downloads weights)...", 0.1)
        for i in range(n):
            log_progress(f"Chatterbox VC: candidate {i+1}/{n}", 0.1 + 0.6 * (i / max(1, n)))
            try:
                _, arr, sr = chatterbox_engine.convert_voice(
                    input_path=input_path, ref_path=ref_path,
                    output_path=None, seed=(None if n == 1 else 1000 + i),
                )
                candidates.append((arr, sr))
            except Exception as e:
                if not candidates:
                    raise gr.Error(f"Chatterbox VC failed: {e}") from e
                log_progress(f"Chatterbox VC candidate {i+1} failed, continuing: {e}")

    elif vc_model == "Seed-VC":
        _ensure_seedvc_engine_hot_reload()
        import seed_vc_engine
        if not seed_vc_engine.is_available():
            raise gr.Error("Seed-VC not installed. Run: pip install seed-vc")
        log_progress("Loading Seed-VC model...", 0.1)
        for i in range(n):
            log_progress(f"Seed-VC: candidate {i+1}/{n}", 0.1 + 0.6 * (i / max(1, n)))
            try:
                _, arr, sr = seed_vc_engine.convert_voice(
                    input_path=input_path, ref_path=ref_path,
                    output_path=None, seed=(None if n == 1 else 2000 + i),
                )
                candidates.append((arr, sr))
            except Exception as e:
                try:
                    seed_vc_engine.unload()
                except Exception:
                    pass
                if not candidates:
                    raise gr.Error(
                        f"Seed-VC failed: {e}\n"
                        "If you use Docker with a bind-mounted repo, this app auto-reloads "
                        "``seed_vc_engine`` when its file changes; otherwise run "
                        "``docker compose restart`` after ``git pull``."
                    ) from e
                log_progress(f"Seed-VC candidate {i+1} failed, continuing: {e}")

    elif vc_model == "Chatterbox TTS":
        import chatterbox_engine
        if not chatterbox_engine.is_available():
            raise gr.Error("Chatterbox not available. Run: pip install chatterbox-tts")
        log_progress(f"Transcribing input audio (Whisper-{whisper_size})...", 0.1)
        full_text, lang_code, _ = tr.transcribe_only(
            audio_path=input_path,
            progress_cb=lambda msg: log_progress(msg),
            model_size=str(whisper_size or "base"),
        )
        if not full_text:
            raise gr.Error("Could not extract text from input audio.")
        # Free Whisper + other engines from VRAM so Chatterbox Multilingual
        # (~6–8 GB) actually fits on the GPU. Without this, MTL OOMs and
        # silently falls back to English-only, which then crashes with a
        # CUDA device-side assert on non-English text.
        log_progress("Freeing GPU memory before Chatterbox load...", 0.45)
        try:
            tr.unload_whisper()
        except Exception as e:
            print(f"[Voice Conversion] unload_whisper failed: {e}", flush=True)
        try:
            device_utils.unload_all(except_modules=("chatterbox_engine",))
        except Exception as e:
            print(f"[Voice Conversion] unload_all failed: {e}", flush=True)
        try:
            _cb_model = chatterbox_engine.get_model()
        except Exception as e:
            raise gr.Error(f"Chatterbox model failed to load: {e}") from e
        if not getattr(_cb_model, "_is_multilingual", False) and (lang_code or "en").lower() != "en":
            raise gr.Error(
                f"Chatterbox loaded the English-only model and the detected language is '{lang_code}'. "
                "Most likely cause: not enough free VRAM for the Multilingual model "
                "(needs ~6–8 GB). Close other GPU apps, or run the app with `--low-vram`/`--cpu`, "
                "or pick another TTS engine for non-English speech."
            )
        log_progress(f"Synthesizing with Chatterbox TTS (lang={lang_code})...", 0.5)
        tmp_out = _write_wav_empty()
        try:
            chatterbox_engine.synthesize(
                text=full_text, language=lang_code, ref_audio_path=ref_path,
                output_path=tmp_out,
                exaggeration=float(cb_exaggeration),
                cfg_weight=float(cb_cfg_weight),
            )
        except Exception as e:
            Path(tmp_out).unlink(missing_ok=True)
            raise gr.Error(f"Chatterbox TTS failed: {e}") from e
        import soundfile as sf

        arr, sr = sf.read(tmp_out, dtype="float32", always_2d=False)
        Path(tmp_out).unlink(missing_ok=True)
        if arr.ndim == 2:
            arr = arr.mean(axis=1)
        candidates.append((arr, int(sr)))

    else:  # OpenVoice
        if not openvoice_available():
            raise gr.Error("OpenVoice is not installed. Run: pip install openvoice-cli")
        log_progress("Starting OpenVoice conversion (chunked to reduce VRAM)...", 0.1)
        try:
            ov_path = openvoice_convert(
                input_path=input_path, ref_path=ref_path,
                output_path=None, device=None,
                tau=float(tau), temperature=float(temperature),
                chunk_duration_sec=_RUNTIME["openvoice_chunk_sec"],
                progress_cb=log_progress,
                use_cpu_fallback_on_oom=True,
            )
        except Exception as e:
            raise gr.Error(f"OpenVoice conversion failed: {e}") from e
        import soundfile as sf

        arr, sr = sf.read(ov_path, dtype="float32", always_2d=False)
        if arr.ndim == 2:
            arr = arr.mean(axis=1)
        candidates.append((arr, int(sr)))

    if not candidates:
        raise gr.Error("All candidates failed — no audio produced.")

    # ── Pick best candidate by speaker similarity ─────────────────────────
    if len(candidates) > 1:
        log_progress(f"Scoring {len(candidates)} candidates by speaker similarity...", 0.75)
        best_idx, best_score = 0, -1.0
        scores: list[float] = []
        for i, (arr, sr) in enumerate(candidates):
            s, _ = _similarity_label(ref_path, arr, sr)
            scores.append(s)
            if s > best_score:
                best_score, best_idx = s, i
        log_progress(f"Candidate scores: {['%.2f' % x for x in scores]}; picked #{best_idx+1}")
        arr, sr = candidates[best_idx]
    else:
        arr, sr = candidates[0]

    # ── Optional OpenVoice tone-color post-step ───────────────────────────
    if apply_openvoice_post and openvoice_available() and engine_used != "OpenVoice":
        log_progress("Applying OpenVoice tone-color post-step...", 0.85)
        src_wav = _write_wav(arr, sr)
        try:
            post_path = openvoice_convert(
                input_path=src_wav, ref_path=ref_path,
                output_path=None, device=None,
                tau=0.55, temperature=1.0,
                chunk_duration_sec=_RUNTIME["openvoice_chunk_sec"],
                progress_cb=lambda msg, _p=None: log_progress(msg),
                use_cpu_fallback_on_oom=True,
            )
            import soundfile as sf

            arr2, sr2 = sf.read(post_path, dtype="float32", always_2d=False)
            if arr2.ndim == 2:
                arr2 = arr2.mean(axis=1)
            arr, sr = arr2, int(sr2)
        except Exception as e:
            log_progress(f"OpenVoice post-step failed ({e}) — keeping base output.")
        finally:
            Path(src_wav).unlink(missing_ok=True)

    final_path = _write_wav(arr, sr)
    score, sim_label = _similarity_label(ref_path, arr, sr)

    # Aggressive VRAM reclaim in --low-vram mode.
    if _RUNTIME.get("low_vram"):
        try:
            import device_utils

            device_utils.unload_all()
        except Exception as e:
            print(f"[run_conversion] post-unload failed: {e}", flush=True)

    log_progress(f"Done. {sim_label}", 1.0)
    return final_path, sim_label


def _write_wav_empty() -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        return tmp.name


def load_from_srt(srt_file):
    """Load segments from an uploaded SRT file (alternative to Step 1 transcription)."""
    if srt_file is None:
        raise gr.Error("Please upload an SRT file.")
    srt_path = str(srt_file)
    if not Path(srt_path).exists():
        raise gr.Error("SRT file not found.")
    content = Path(srt_path).read_text(encoding="utf-8", errors="replace")
    segments_text, total_duration = tr.parse_srt(content)
    if not segments_text:
        raise gr.Error("No segments found in the SRT file. Check that it is a valid SRT.")
    return segments_text, total_duration


def run_polish_with_ai(segments_text, target_language, context_hint, api_key, progress=gr.Progress()):
    """Polish/fix segments using Claude AI."""
    if not segments_text or not segments_text.strip():
        raise gr.Error("No segments to polish. Run Step 1 or load an SRT first.")
    lang_code = tr.LANGUAGES[target_language]["lang_code"]
    progress(0.2, desc="Sending segments to Claude AI...")
    try:
        result = tr.polish_segments_with_ai(
            segments_text=segments_text,
            lang_code=lang_code,
            context_hint=context_hint,
            api_key=api_key,
        )
    except ValueError as e:
        raise gr.Error(str(e))
    except Exception as e:
        raise gr.Error(f"AI polishing failed: {e}")
    progress(1.0, desc="Done.")
    return result


def run_step1_transcribe(
    input_audio, target_language, female_narrator,
    progress=gr.Progress(),
):
    """Tab 2 Step 1: Whisper ASR → translate → return editable segment text."""
    input_path = _audio_path(input_audio)
    if not input_path or not Path(input_path).exists():
        raise gr.Error("Please upload an input audio file.")

    step_count = [0]

    def log(msg: str):
        step_count[0] += 1
        progress(min(0.05 + step_count[0] * 0.08, 0.95), desc=msg)

    segments_text, total_duration = tr.transcribe_and_translate(
        audio_path=input_path,
        target_language=target_language,
        female_narrator=bool(female_narrator),
        progress_cb=log,
    )
    progress(1.0, desc="Done. Edit segments below, then click Synthesize.")
    return segments_text, total_duration, input_path


def run_step2_synthesize(
    edited_text, total_duration, reference_audio, target_language,
    topk, sync_mode, tts_rate_pct, tts_pitch_hz,
    tts_backend, xtts_speed,
    chatterbox_exaggeration, chatterbox_cfg_weight,
    f5tts_ref_text, f5tts_model_path, f5tts_speed,
    input_audio_path,
    preprocess_reference: bool = True,
    best_of_n: int = 1,
    apply_openvoice_post: bool = False,
    progress=gr.Progress(),
):
    """Tab 2 Step 2: TTS + optional voice conversion from edited segments."""
    if not edited_text or not edited_text.strip():
        raise gr.Error("No segments to synthesize. Run Step 1 first.")
    total_duration = float(total_duration or 0)
    if total_duration <= 0:
        total_duration = tr.infer_duration_from_segments(edited_text)
    if total_duration <= 0:
        raise gr.Error(
            "Missing audio duration. Run Step 1 first, or use segment lines like [0.00 - 2.50] text."
        )

    ref_path_raw = _audio_path(reference_audio) if reference_audio else None
    needs_ref = tts_backend in ("XTTS v2", "Chatterbox", "F5-TTS", "Edge-TTS + OpenVoice")
    use_openvoice = tts_backend == "Edge-TTS + OpenVoice"

    if needs_ref and (not ref_path_raw or not Path(ref_path_raw).exists()):
        raise gr.Error(
            f"„{tts_backend}” wymaga **osobnej** próbki głosu docelowego (Reference Voice Sample, 10–30 s). "
            "To nie może być to samo audio co wejście z kroku 1 — tam jest stary głos / treść do przetłumaczenia. "
            "Przygotuj plik w zakładce „Przygotuj referencję” i wgraj go tutaj."
        )

    ref_path = ref_path_raw
    ref_status = ""
    if ref_path and preprocess_reference:
        ref_path, ref_status = _prepared_ref_path(ref_path_raw, preprocess=True, denoise=True)

    step_count = [0]
    status_lines: list[str] = [f"**Wybrany silnik:** {tts_backend}"]
    if ref_status:
        status_lines.append(f"**Referencja:** {ref_status}")

    def log(msg: str):
        step_count[0] += 1
        status_lines.append(msg)
        progress(min(0.1 + step_count[0] * 0.07, 0.92), desc=msg)

    if use_openvoice:
        status_lines.append(
            "**Uwaga:** Edge-TTS + OpenVoice daje słabe podobieństwo do referencji "
            "(najpierw głos Microsoftu, potem tylko kosmetyka barwy). "
            "Dla prawdziwego klonu wybierz **Chatterbox**."
        )

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out_path = tmp.name
    post_path = out_path

    try:
        output_sr = SAMPLE_RATE
        synth_kwargs = dict(
            edited_text=edited_text,
            total_duration=float(total_duration),
            target_language=target_language,
            sync=sync_mode or "off",
            xtts_ref_path=ref_path,
            chatterbox_exaggeration=float(chatterbox_exaggeration),
            chatterbox_cfg_weight=float(chatterbox_cfg_weight),
            f5tts_ref_text=f5tts_ref_text or "",
            f5tts_speed=float(f5tts_speed),
            best_of_n=max(1, int(best_of_n or 1)),
            progress_cb=log,
        )

        if tts_backend == "XTTS v2":
            progress(0.05, desc="Loading XTTS v2 model (first run: downloads ~1.8 GB)...")
            output_sr = tr.synthesize_from_edited(
                output_path=out_path,
                tts_backend="xtts",
                xtts_speed=float(xtts_speed),
                **synth_kwargs,
            )

        elif tts_backend == "Chatterbox":
            progress(0.05, desc="Loading Chatterbox Multilingual (wymaga referencji + VRAM)...")
            output_sr = tr.synthesize_from_edited(
                output_path=out_path,
                tts_backend="chatterbox",
                **synth_kwargs,
            )

        elif tts_backend == "F5-TTS":
            model_path = f5tts_model_path.strip() if f5tts_model_path else None
            if model_path == "":
                model_path = None
            if model_path is None:
                lang_meta = tr.LANGUAGES.get(target_language, {})
                if lang_meta.get("lang_code") == "pl":
                    model_path = "polish"
            progress(0.05, desc="Loading F5-TTS model...")
            output_sr = tr.synthesize_from_edited(
                output_path=out_path,
                tts_backend="f5tts",
                f5tts_model_path=model_path,
                **synth_kwargs,
            )

        else:
            rate_str = f"{int(tts_rate_pct):+d}%"
            pitch_str = f"{int(tts_pitch_hz):+d}Hz"

            if use_openvoice and not openvoice_available():
                raise gr.Error(
                    "Wybrano „Edge-TTS + OpenVoice”, ale OpenVoice nie jest zainstalowany. "
                    "W katalogu projektu uruchom: `.venv/bin/pip install openvoice-cli` "
                    "i zrestartuj aplikację. Alternatywa: wybierz **Chatterbox**."
                )

            if use_openvoice:
                import tempfile as _tf
                with _tf.NamedTemporaryFile(suffix=".wav", delete=False) as _tmp:
                    tts_only_path = _tmp.name
                output_sr = tr.synthesize_from_edited(
                    output_path=tts_only_path,
                    tts_rate=rate_str,
                    tts_pitch=pitch_str,
                    tts_backend="edge",
                    **synth_kwargs,
                )
                progress(0.85, desc="Applying OpenVoice tone-color transfer...")
                openvoice_convert(
                    input_path=tts_only_path,
                    ref_path=ref_path,
                    output_path=out_path,
                    chunk_duration_sec=_RUNTIME["openvoice_chunk_sec"],
                    progress_cb=lambda msg, _p=None: log(msg),
                )
                Path(tts_only_path).unlink(missing_ok=True)
            else:
                output_sr = tr.synthesize_from_edited(
                    output_path=out_path,
                    tts_rate=rate_str,
                    tts_pitch=pitch_str,
                    tts_backend="edge",
                    **synth_kwargs,
                )

        if (
            apply_openvoice_post
            and ref_path
            and tts_backend in ("Chatterbox", "F5-TTS", "XTTS v2")
            and openvoice_available()
        ):
            progress(0.9, desc="OpenVoice post-pass (timbre unify)...")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as post_tmp:
                post_path = post_tmp.name
            _apply_openvoice_post_mix(out_path, ref_path, post_path, tau=0.4, progress_cb=log)
            Path(out_path).unlink(missing_ok=True)
            out_path = post_path

        out_wav, _ = librosa.load(out_path, sr=output_sr, mono=True)
    except RuntimeError as e:
        raise gr.Error(str(e)) from e
    finally:
        Path(out_path).unlink(missing_ok=True)

    progress(1.0, desc="Done.")
    status_lines.append(
        "W logach terminala szukaj linii `Silnik: Chatterbox` lub `Chatterbox 1/N` — "
        "jeśli widzisz `Edge-TTS`, klonowanie nie działało."
    )
    return (output_sr, out_wav), "\n\n".join(status_lines)


def build_ui():
    with gr.Blocks(title="Voice Changer") as demo:
        gr.Markdown("# Voice Changer — Zero-Shot")

        has_cb_tts = _has_chatterbox_tts()
        has_cb_vc = _has_chatterbox_vc()
        has_ov = bool(openvoice_available())
        has_xtts = _has_xtts()
        has_f5 = _has_f5tts()
        has_bark = _has_bark()
        has_seedvc = _has_seedvc()
        has_ss = _has_speaker_sim()

        with gr.Tabs():
            # ── Tab 0: Tagged TTS (text-only) ───────────────────────────────
            with gr.Tab("Tagged TTS"):
                gr.Markdown(
                    "Wklej tekst (bez timestampów) z tagami typu `[śmiech]`, `[westchnienie]`, `[pauza]`.\n\n"
                    "- **Mowa**: synteza przez wybrany silnik (Chatterbox/XTTS/F5/Edge).\n"
                    "- **Tagi niewerbalne**: opcjonalnie Bark (lokalnie), żeby śmiech/westchnienie było w WAV.\n\n"
                    "**Tip:** możesz też wgrać WAV i kliknąć „Transcribe sample → text”, żeby wypełnić pole tekstu."
                )

                with gr.Row():
                    with gr.Column():
                        tag_ref = gr.Audio(
                            label="Reference Voice Sample (opcjonalnie — do klonowania głosu mowy)",
                            type="filepath",
                        )
                        tag_transcribe_btn = gr.Button("Transcribe sample → text", variant="secondary")
                        tag_text = gr.Textbox(
                            label="Script (no timestamps)",
                            lines=10,
                            placeholder="No cześć [delikatny śmiech] miło cię widzieć.",
                        )
                        with gr.Row():
                            speech_engine = gr.Dropdown(
                                choices=["F5-TTS", "Chatterbox", "XTTS v2", "Edge-TTS"],
                                value=("F5-TTS" if has_f5 else ("Chatterbox" if has_cb_tts else "Edge-TTS")),
                                label="Speech engine",
                            )
                            use_bark_tags = gr.Checkbox(
                                value=False,
                                label="Use Bark for non-speech tags ([śmiech]/[westchnienie]) — optional",
                            )
                        bark_preset = gr.Textbox(
                            label="Bark preset (history_prompt)",
                            value="v2/pl_speaker_0",
                            info="Używane tylko do efektów tagów. Instalacja: pip install git+https://github.com/suno-ai/bark.git",
                        )
                        render_btn = gr.Button("Generate audio", variant="primary")

                    with gr.Column():
                        tag_out = gr.Audio(label="Output audio", type="numpy")

                def _transcribe_ref(ref_audio, progress=gr.Progress()):
                    ref_path = _audio_path(ref_audio)
                    if not ref_path or not Path(ref_path).exists():
                        raise gr.Error("Upload a WAV/MP3 first.")
                    progress(0.1, desc="Transcribing with Whisper…")
                    text, _lang, _dur = tr.transcribe_only(ref_path)
                    progress(1.0, desc="Done.")
                    return text

                def _render_tagged(script, ref_audio, engine, bark_on, bark_hp, progress=gr.Progress()):
                    script = script or ""
                    ref_path = _audio_path(ref_audio)
                    lang = "pl"
                    progress(0.1, desc="Rendering…")
                    speech = ("chatterbox" if engine == "Chatterbox" else
                              "xtts" if engine == "XTTS v2" else
                              "f5tts" if engine == "F5-TTS" else
                              "edge")
                    # Graceful fallback: if selected heavy backend is unavailable, use Edge-TTS.
                    if speech == "chatterbox" and not has_cb_tts:
                        progress(0.2, desc="Chatterbox unavailable → falling back to Edge-TTS")
                        speech = "edge"
                    elif speech == "xtts" and not has_xtts:
                        progress(0.2, desc="XTTS unavailable → falling back to Edge-TTS")
                        speech = "edge"
                    elif speech == "f5tts" and not has_f5:
                        progress(0.2, desc="F5-TTS unavailable → falling back to Edge-TTS")
                        speech = "edge"

                    if bark_on and not has_bark:
                        progress(0.2, desc="Bark not installed → rendering without non-speech Bark tags")
                        bark_on = False
                    try:
                        wav, out_sr = tagged_tts.render_tagged_script(
                            script,
                            speech_engine=speech,
                            language=lang,
                            ref_audio_path=ref_path,
                            use_bark_for_tags=bool(bark_on),
                            bark_preset=bark_hp or "v2/pl_speaker_0",
                            f5tts_model_path="polish" if speech == "f5tts" else None,
                        )
                    except RuntimeError as e:
                        raise gr.Error(str(e)) from e
                    progress(1.0, desc="Done.")
                    return (out_sr, wav)

                tag_transcribe_btn.click(
                    fn=_transcribe_ref,
                    inputs=[tag_ref],
                    outputs=[tag_text],
                )
                render_btn.click(
                    fn=_render_tagged,
                    inputs=[tag_text, tag_ref, speech_engine, use_bark_tags, bark_preset],
                    outputs=[tag_out],
                )

            # ── Tab 1: Voice Conversion ──────────────────────────────────────
            with gr.Tab("Voice Conversion"):
                gr.Markdown(
                    "Skonwertuj głos z nagrania wejściowego na głos z próbki referencyjnej.\n\n"
                    "**Tryby:** **Chatterbox VC** (resynteza przez tokenizer S3, zachowuje prozodię), "
                    "**Seed-VC** (SOTA zero-shot identity transfer), "
                    "**OpenVoice** (transfer tonu), "
                    "**Chatterbox TTS** (transkrypcja → TTS z klonowaniem głosu — traci prozodię)."
                )
                if not (has_cb_vc or has_cb_tts or has_ov or has_seedvc):
                    gr.Markdown(
                        "⚠️ **Brak dostępnych backendów Voice Conversion w tym środowisku.**\n\n"
                        "- Chatterbox: `pip install chatterbox-tts`\n"
                        "- Seed-VC: `pip install seed-vc`\n"
                        "- OpenVoice: `pip install openvoice-cli`"
                    )
                _vc_choices = [c for c, ok in [
                    ("Chatterbox VC", has_cb_vc),
                    ("Seed-VC", has_seedvc),
                    ("OpenVoice", has_ov),
                    ("Chatterbox TTS", has_cb_tts),
                ] if ok] or ["Chatterbox VC"]
                _vc_default = _vc_choices[0]

                with gr.Row():
                    with gr.Column():
                        vc_input = gr.Audio(label="Input Audio", type="filepath")
                        vc_ref = gr.Audio(
                            label="Reference Voice Sample (5–30 s)", type="filepath"
                        )
                        vc_model_radio = gr.Radio(
                            choices=_vc_choices,
                            value=_vc_default,
                            label="Conversion Model",
                        )
                        with gr.Group(visible=(_vc_default == "Chatterbox VC")) as vc_group_chatterbox_vc:
                            gr.Markdown(
                                "**Chatterbox VC** — resyntezuje audio zachowując treść, "
                                "aplikując tembr głosu z próbki referencyjnej (tokenizer S3). "
                                "Stochastyczny → best-of-N pomaga."
                            )
                        with gr.Group(visible=(_vc_default == "Seed-VC")) as vc_group_seedvc:
                            gr.Markdown(
                                "**Seed-VC** — SOTA zero-shot VC z silnym transferem tożsamości. "
                                "Dobrze reaguje na `best-of-N` (różny seed)."
                            )
                        with gr.Group(visible=(_vc_default == "Chatterbox TTS")) as vc_group_chatterbox_tts:
                            gr.Markdown(
                                "**Chatterbox TTS** — transkrypcja Whisper → synteza TTS z klonowaniem głosu. "
                                "Uwaga: traci prozodię źródła."
                            )
                            vc_cb_exaggeration = gr.Slider(
                                minimum=0.0, maximum=1.0, value=0.5, step=0.05,
                                label="Emotion exaggeration",
                                info="Higher = more expressive/dramatic speech.",
                            )
                            vc_cb_cfg = gr.Slider(
                                minimum=0.0, maximum=1.0, value=0.5, step=0.05,
                                label="CFG weight (guidance)",
                                info="Higher = more faithful to reference voice.",
                            )
                            vc_whisper_size = gr.Dropdown(
                                choices=["base", "small", "medium", "large-v2", "large-v3"],
                                value="base",
                                label="Whisper model size",
                                info="Większy model = lepsza transkrypcja źródła → lepsza re-synteza. "
                                     "`large-v3` wymaga GPU dla rozsądnego czasu.",
                            )
                        with gr.Group(visible=(_vc_default == "OpenVoice")) as vc_group_openvoice:
                            gr.Markdown("**OpenVoice settings**")
                            vc_tau = gr.Slider(
                                minimum=0.3, maximum=1.0, value=0.65, step=0.05,
                                label="Reference voice strength (tau)",
                                info="Higher = output closer to reference.",
                            )
                            vc_temperature = gr.Slider(
                                minimum=0.7, maximum=1.3, value=1.0, step=0.05,
                                label="Voice temperature",
                                info="Higher = warmer, more expressive; lower = calmer.",
                            )

                        with gr.Accordion("Advanced (reference preprocessing, best-of-N, post)", open=False):
                            vc_preprocess = gr.Checkbox(
                                value=True,
                                label="Preprocess reference (denoise / LUFS / VAD / best-window)",
                                info="Włączone zalecane. Wynik jest cache’owany obok pliku referencyjnego.",
                            )
                            vc_denoise = gr.Checkbox(
                                value=True,
                                label="Denoise reference (noisereduce)",
                                info="Usuń szum tła z próbki (wymaga `pip install noisereduce`).",
                            )
                            vc_best_of_n = gr.Slider(
                                minimum=1, maximum=5, value=(3 if has_ss else 1), step=1,
                                label="Best-of-N candidates",
                                info=(
                                    "Liczba prób (Chatterbox VC / Seed-VC losują). "
                                    "Najlepsza wybierana po podobieństwie mówcy." if has_ss else
                                    "Similarity nieaktywne (brak speechbrain/resemblyzer) → >1 nie przyniesie poprawy."
                                ),
                            )
                            vc_apply_ov_post = gr.Checkbox(
                                value=False,
                                label="Apply OpenVoice tone-color as post-step",
                                info="Dodatkowy transfer tonu po głównej konwersji (wymaga OpenVoice).",
                                visible=has_ov,
                            )

                        if not has_ss:
                            gr.Markdown(
                                "ℹ️ Pomiar podobieństwa mówcy niedostępny — `pip install speechbrain` "
                                "lub `pip install resemblyzer`, żeby włączyć scoring i best-of-N."
                            )

                        vc_btn = gr.Button("Convert Voice", variant="primary")
                    with gr.Column():
                        vc_output = gr.Audio(label="Converted Audio", type="filepath")
                        vc_similarity = gr.Textbox(
                            label="Speaker similarity",
                            value="",
                            interactive=False,
                            info=">=0.72 = dobry klon, 0.60–0.72 = akceptowalny, <0.55 = słaby (zmień referencję).",
                        )

                def _update_vc_model_ui(model):
                    return (
                        gr.update(visible=model == "Chatterbox VC"),
                        gr.update(visible=model == "Seed-VC"),
                        gr.update(visible=model == "Chatterbox TTS"),
                        gr.update(visible=model == "OpenVoice"),
                    )

                def _on_model_change(model):
                    """Free VRAM of the previously-active VC engine."""
                    try:
                        import device_utils

                        keep_map = {
                            "Chatterbox VC":  ("chatterbox_engine",),
                            "Chatterbox TTS": ("chatterbox_engine",),
                            "Seed-VC":        ("seed_vc_engine",),
                            "OpenVoice":      ("openvoice_engine",),
                        }
                        device_utils.unload_all(except_modules=keep_map.get(model, ()))
                    except Exception as e:
                        print(f"[vc_model_radio.change] unload_all failed: {e}", flush=True)
                    return _update_vc_model_ui(model)

                vc_model_radio.change(
                    fn=_on_model_change,
                    inputs=[vc_model_radio],
                    outputs=[vc_group_chatterbox_vc, vc_group_seedvc,
                             vc_group_chatterbox_tts, vc_group_openvoice],
                )
                vc_btn.click(
                    fn=run_conversion,
                    inputs=[
                        vc_input, vc_ref, vc_model_radio,
                        vc_cb_exaggeration, vc_cb_cfg,
                        vc_tau, vc_temperature,
                        vc_preprocess, vc_denoise, vc_best_of_n, vc_apply_ov_post,
                        vc_whisper_size,
                    ],
                    outputs=[vc_output, vc_similarity],
                )
                gr.Markdown(
                    "**Tips:** Referencja 15–25 s, czysta mowa, jeden głos. "
                    "Najlepsza jakość klonu: **Chatterbox VC** lub **Seed-VC** z preprocessingiem i best-of-N=3. "
                    "Włącz „Apply OpenVoice post”, jeśli chcesz dodatkowo dociągnąć tembr."
                )

            # ── Tab: Prepare Reference ───────────────────────────────────────
            with gr.Tab("Przygotuj referencję"):
                gr.Markdown(
                    "Wgraj **dowolnie długie** nagranie z głosem docelowym. "
                    "Aplikacja wycina ciszę i fragmenty bez mowy (VAD), opcjonalnie redukuje szum, "
                    "normalizuje głośność i wybiera **najczystsze okno** (~8–25 s) pod klonowanie głosu.\n\n"
                    "Wynik zapisuje się obok pliku źródłowego jako `*.ref<hash>.wav` (24 kHz, mono) "
                    "oraz można go pobrać poniżej."
                )
                with gr.Row():
                    with gr.Column():
                        refprep_input = gr.Audio(
                            label="Nagranie źródłowe (dowolna długość)",
                            type="filepath",
                        )
                        refprep_seconds = gr.Slider(
                            minimum=8,
                            maximum=25,
                            value=20,
                            step=1,
                            label="Docelowa długość referencji (s)",
                            info="Z najczystszego fragmentu wycina okno o tej długości.",
                        )
                        refprep_denoise = gr.Checkbox(
                            value=True,
                            label="Redukcja szumu (noisereduce)",
                        )
                        refprep_cache = gr.Checkbox(
                            value=True,
                            label="Użyj cache (szybsze ponowne przetwarzanie tego samego pliku)",
                        )
                        refprep_btn = gr.Button(
                            "Przygotuj plik referencyjny",
                            variant="primary",
                        )
                    with gr.Column():
                        refprep_output = gr.Audio(
                            label="Gotowa referencja (24 kHz)",
                            type="filepath",
                        )
                        refprep_report = gr.Markdown(label="Raport")
                refprep_btn.click(
                    fn=run_prepare_reference,
                    inputs=[refprep_input, refprep_seconds, refprep_denoise, refprep_cache],
                    outputs=[refprep_output, refprep_report],
                )
                gr.Markdown(
                    "**Wskazówki:** jeden mówca, bez muzyki w tle; im więcej czystej mowy w źródle, "
                    "tym lepszy wybór okna. Do Chatterbox / Seed-VC wklej ten plik jako referencję "
                    "(preprocessing w Voice Conversion możesz wtedy wyłączyć — plik jest już gotowy)."
                )

            # ── Tab 2: Translate & Convert ───────────────────────────────────
            with gr.Tab("Translate & Convert"):
                gr.Markdown(
                    "Translate speech from **any language** to a target language, "
                    "then optionally apply voice conversion to match a reference speaker.\n\n"
                    "**Step 1:** Transcribe & translate → review/edit segments → "
                    "**Step 2:** Synthesize & convert"
                )
                # Shared state: total audio duration + input audio path (for XTTS auto-reference)
                tr_duration = gr.State(0.0)
                tr_input_state = gr.State(None)  # stores input audio path after step 1

                with gr.Row():
                    # ── Step 1 inputs ────────────────────────────────────────
                    with gr.Column():
                        gr.Markdown("### Step 1 — Transcribe & Translate")
                        tr_input = gr.Audio(label="Input Audio (any language)", type="filepath")
                        tr_lang = gr.Dropdown(
                            choices=list(tr.LANGUAGES.keys()),
                            value="Polish",
                            label="Target Language",
                        )
                        tr_female = gr.Checkbox(
                            value=True,
                            label="Female narrator",
                            info="Use feminine grammatical forms in the translation",
                        )
                        tr_step1_btn = gr.Button("Transcribe & Translate", variant="primary")
                        gr.Markdown("**— or load from SRT —**")
                        tr_srt = gr.File(label="SRT File", file_types=[".srt"], type="filepath")
                        tr_srt_btn = gr.Button("Load SRT")

                    # ── Editable segments ────────────────────────────────────
                    with gr.Column():
                        gr.Markdown("### Translated Segments (editable)")
                        tr_segments = gr.Textbox(
                            label="Segments — format: [start - end] text",
                            lines=12,
                            interactive=True,
                            placeholder="Click 'Transcribe & Translate' to populate this field.\n"
                                        "Then edit any segment text before synthesizing.",
                        )

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### AI Segment Polishing (optional)")
                        tr_context = gr.Textbox(
                            label="Narrative context / style hint",
                            placeholder="e.g. Monologue of narrator to listener. English language.",
                            lines=2,
                        )
                        tr_api_key = gr.Textbox(
                            label="Anthropic API Key (or set ANTHROPIC_API_KEY env var)",
                            placeholder="sk-ant-...",
                            type="password",
                        )
                        tr_polish_btn = gr.Button("Polish with AI", variant="secondary")

                with gr.Row():
                    # ── Step 2 inputs ────────────────────────────────────────
                    with gr.Column():
                        gr.Markdown("### Step 2 — Synthesize & Convert")
                        tr_tts_backend = gr.Radio(
                            choices=["Edge-TTS + OpenVoice", "XTTS v2", "Chatterbox", "F5-TTS"],
                            value=_default_tts_backend(has_f5, has_cb_tts),
                            label="TTS Engine",
                            info=(
                                "F5-TTS: najlepsza naturalność PL (checkpoint polish). "
                                "Chatterbox: zero-shot, natywny polski. "
                                "XTTS v2: zero-shot klonowanie głosu. "
                                "Edge-TTS + OpenVoice: lekka opcja online."
                            ),
                        )
                        tr_ref = gr.Audio(
                            label="Reference Voice Sample — głos DOCELOWY (10–30 s, nie audio z kroku 1)",
                            type="filepath",
                        )
                        tr_ref_preprocess = gr.Checkbox(
                            value=True,
                            label="Przygotuj referencję przed syntezą (VAD / denoise / najlepsze okno)",
                        )
                        tr_sync = gr.Radio(
                            choices=["off", "gentle", "strict"],
                            value="off",
                            label="Timing sync mode",
                            info=(
                                "off = naturalna mowa (zalecane). "
                                "gentle = dopasuj pauzy bez przyspieszania. "
                                "strict = wymuś sloty SRT (może ścinać/przyspieszać)."
                            ),
                        )
                        tr_best_of_n = gr.Slider(
                            minimum=1, maximum=5, value=1, step=1,
                            label="Best-of-N (TTS)",
                            info="Generuj N wariantów i wybierz najwyższe podobieństwo do referencji (wolniejsze).",
                        )
                        tr_openvoice_post = gr.Checkbox(
                            value=False,
                            label="OpenVoice post-pass (unify timbre across segments)",
                            info="Lekki pass OpenVoice na całym mixie — wymaga openvoice-cli.",
                        )
                        with gr.Group(visible=True) as tr_group_chatterbox:
                            gr.Markdown(
                                "**Chatterbox settings** — long text is auto-split into ~220 char chunks. "
                                "Ucięte słowa? Wyłącz sync lub skróć tekst w segmencie."
                            )
                            tr_cb_exaggeration = gr.Slider(
                                minimum=0.0, maximum=1.0, value=0.35, step=0.05,
                                label="Emotion exaggeration",
                                info="Lower = more natural narration (0.3–0.4 for Polish).",
                            )
                            tr_cb_cfg = gr.Slider(
                                minimum=0.0, maximum=1.0, value=0.7, step=0.05,
                                label="CFG weight (guidance)",
                                info="Wyżej = bliżej referencji (0.7–0.85). Za nisko brzmi jak głos bazowy.",
                            )
                        with gr.Group(visible=False) as tr_group_f5tts:
                            gr.Markdown("**F5-TTS settings**")
                            tr_f5_ref_text = gr.Textbox(
                                label="Reference audio transcript (F5-TTS)",
                                placeholder="Wpisz transkrypt nagrania referencyjnego dla lepszej jakości...",
                                lines=2,
                                info="Transcript of the reference voice audio. Leave empty to auto-detect.",
                            )
                            tr_f5_model_path = gr.Textbox(
                                label="F5-TTS model path (optional)",
                                value="polish",
                                placeholder='Puste = model bazowy. Wpisz "polish" dla modelu PL lub ścieżkę do .pt',
                                info='Domyślnie "polish" — community checkpoint PL (~3 GB).',
                            )
                            tr_f5_speed = gr.Slider(
                                minimum=0.5, maximum=2.0, value=1.0, step=0.05,
                                label="F5-TTS Speaking Speed",
                            )
                        with gr.Group(visible=False) as tr_group_xtts:
                            gr.Markdown("**XTTS v2 settings**")
                            tr_xtts_speed = gr.Slider(
                                minimum=0.5, maximum=2.0, value=1.0, step=0.05,
                                label="XTTS Speaking Speed",
                            )
                        with gr.Group(visible=False) as tr_group_edge:
                            gr.Markdown("**Edge-TTS settings**")
                            tr_topk = gr.Slider(
                                minimum=1, maximum=16, value=4, step=1,
                                label="Top-K Neighbors (kNN-VC only)",
                            )
                            tr_rate = gr.Slider(
                                minimum=-30, maximum=30, value=0, step=1,
                                label="TTS Speaking Rate %",
                                info="Negative = slower, positive = faster.",
                            )
                            tr_pitch = gr.Slider(
                                minimum=-20, maximum=20, value=0, step=1,
                                label="TTS Pitch Hz",
                                info="Negative = deeper, positive = higher.",
                            )
                        tr_step2_btn = gr.Button("Synthesize & Convert", variant="primary")

                    # ── Output ───────────────────────────────────────────────
                    with gr.Column():
                        gr.Markdown("### Output")
                        tr_output = gr.Audio(label="Output Audio", type="numpy")
                        tr_step2_status = gr.Markdown(
                            label="Status syntezy",
                            value="Po syntezie tutaj pojawi się użyty silnik i podpowiedzi.",
                        )

                tr_step1_btn.click(
                    fn=run_step1_transcribe,
                    inputs=[tr_input, tr_lang, tr_female],
                    outputs=[tr_segments, tr_duration, tr_input_state],
                )
                tr_srt_btn.click(
                    fn=load_from_srt,
                    inputs=[tr_srt],
                    outputs=[tr_segments, tr_duration],
                )
                tr_polish_btn.click(
                    fn=run_polish_with_ai,
                    inputs=[tr_segments, tr_lang, tr_context, tr_api_key],
                    outputs=[tr_segments],
                )
                def _update_backend_ui(backend):
                    return (
                        gr.update(visible=backend == "Chatterbox"),
                        gr.update(visible=backend == "F5-TTS"),
                        gr.update(visible=backend == "XTTS v2"),
                        gr.update(visible=backend == "Edge-TTS + OpenVoice"),
                    )

                tr_tts_backend.change(
                    fn=_update_backend_ui,
                    inputs=[tr_tts_backend],
                    outputs=[tr_group_chatterbox, tr_group_f5tts, tr_group_xtts, tr_group_edge],
                )

                tr_step2_btn.click(
                    fn=run_step2_synthesize,
                    inputs=[
                        tr_segments, tr_duration, tr_ref, tr_lang,
                        tr_topk, tr_sync, tr_rate, tr_pitch,
                        tr_tts_backend, tr_xtts_speed,
                        tr_cb_exaggeration, tr_cb_cfg,
                        tr_f5_ref_text, tr_f5_model_path, tr_f5_speed,
                        tr_input_state,
                        tr_ref_preprocess,
                        tr_best_of_n,
                        tr_openvoice_post,
                    ],
                    outputs=[tr_output, tr_step2_status],
                )
                gr.Markdown(
                    "**Klonowanie:** wybierz **Chatterbox** i wgraj osobną referencję. "
                    "W terminalu muszą być linie `Chatterbox 1/N`, nie `Edge-TTS`. "
                    "**Edge-TTS + OpenVoice** nie daje pełnego klonu. "
                    "**F5-TTS:** 5–15 s referencji + opcjonalnie transkrypt; ścieżka `polish` dla PL."
                )



    return demo


def main():
    parser = argparse.ArgumentParser(description="Voice Conversion Web UI")
    parser.add_argument("--port", type=int, default=7862, help="Port to listen on")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument(
        "--low-vram", action="store_true",
        help="Reduce GPU memory use: kNN-VC / WavLM chunks, smaller OpenVoice time chunks (~10 s).",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Do not use NVIDIA GPUs (CUDA hidden). Chatterbox/OpenVoice run on CPU — slower, no VRAM errors.",
    )
    args = parser.parse_args()

    if args.cpu:
        os.environ.setdefault("VOICE_CHANGER_FORCE_DEVICE", "cpu")
        print("CPU mode: CUDA disabled for this process (--cpu).")

    if args.low_vram:
        import voice_utils
        voice_utils.CHUNK_SECONDS = 5
        voice_utils.MATCH_CHUNK_FRAMES = 300
        _RUNTIME["openvoice_chunk_sec"] = min(10.0, float(_OV_DEFAULT_CHUNK_SEC))
        _RUNTIME["low_vram"] = True
        print(
            "Low-VRAM mode: kNN feature chunks=5s, vocoder chunks=300 frames; "
            f"OpenVoice chunk length={_RUNTIME['openvoice_chunk_sec']:.0f}s; "
            "engines auto-unloaded after each conversion."
        )

    # kNN-VC is loaded only when using "Translate & Convert" with reference (saves ~9.5 GB VRAM for OpenVoice in Tab 1)

    demo = build_ui()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share, inbrowser=True)


if __name__ == "__main__":
    main()
