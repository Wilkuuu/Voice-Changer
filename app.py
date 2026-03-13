#!/usr/bin/env python3
"""
Gradio web interface for natural voice conversion (OpenVoice) and translation pipeline.

Voice Conversion tab: OpenVoice tone-color transfer (GPU, natural cloning).
Translate & Convert tab: Whisper + TTS + optional kNN-VC.

Usage:
    python app.py
    python app.py --share   # public URL via Gradio tunnel
    python app.py --port 7861
"""

import argparse
import os
import tempfile
from pathlib import Path

# Must be set before CUDA initializes — reduces OOM from memory fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gradio as gr
import librosa
import torch

import translate as tr

try:
    from openvoice_engine import is_available as openvoice_available, convert as openvoice_convert
except ImportError:
    openvoice_available = lambda: False
    openvoice_convert = None

SAMPLE_RATE = 16000
_model = None


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


def run_conversion(
    input_audio, reference_audio, vc_model,
    cb_exaggeration, cb_cfg_weight,
    tau, temperature,
    progress=gr.Progress(),
):
    """Tab 1: Voice conversion — Chatterbox VC, Chatterbox TTS, or OpenVoice."""
    input_path = _audio_path(input_audio)
    ref_path = _audio_path(reference_audio)
    if not input_path or not Path(input_path).exists():
        raise gr.Error("Please upload an input audio file.")
    if not ref_path or not Path(ref_path).exists():
        raise gr.Error("Please upload a reference voice sample.")

    def log_progress(msg: str, p: float | None = None) -> None:
        print(f"[Voice Conversion] {msg}", flush=True)
        if p is not None:
            progress(float(p), desc=msg)
        else:
            progress(None, desc=msg)

    if vc_model == "Chatterbox VC":
        import chatterbox_engine
        if not chatterbox_engine.is_vc_available():
            raise gr.Error("Chatterbox VC not available. Run: pip install chatterbox-tts")
        log_progress("Loading Chatterbox VC model (first run downloads weights)...", 0.0)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_path = tmp.name
        try:
            chatterbox_engine.convert_voice(
                input_path=input_path,
                ref_path=ref_path,
                output_path=out_path,
            )
        except Exception as e:
            Path(out_path).unlink(missing_ok=True)
            raise gr.Error(f"Chatterbox VC failed: {e}") from e
        log_progress("Done.", 1.0)
        return out_path

    elif vc_model == "Chatterbox TTS":
        import chatterbox_engine
        if not chatterbox_engine.is_available():
            raise gr.Error("Chatterbox not available. Run: pip install chatterbox-tts")
        log_progress("Transcribing input audio with Whisper...", 0.1)
        # Transcribe input to get the text
        segments_text, _ = tr.transcribe_and_translate(
            audio_path=input_path,
            target_language="English",
            female_narrator=False,
            progress_cb=lambda msg: log_progress(msg),
        )
        # Extract plain text from segments
        lines = []
        for line in segments_text.strip().splitlines():
            line = line.strip()
            if line.startswith("[") and "]" in line:
                text_part = line.split("]", 1)[1].strip()
                if text_part:
                    lines.append(text_part)
        full_text = " ".join(lines)
        if not full_text:
            raise gr.Error("Could not extract text from input audio.")
        log_progress("Synthesizing with Chatterbox TTS...", 0.5)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_path = tmp.name
        try:
            chatterbox_engine.synthesize(
                text=full_text,
                language="en",
                ref_audio_path=ref_path,
                output_path=out_path,
                exaggeration=float(cb_exaggeration),
                cfg_weight=float(cb_cfg_weight),
            )
        except Exception as e:
            Path(out_path).unlink(missing_ok=True)
            raise gr.Error(f"Chatterbox TTS failed: {e}") from e
        log_progress("Done.", 1.0)
        return out_path

    else:  # OpenVoice
        if not openvoice_available():
            raise gr.Error("OpenVoice is not installed. Run: pip install openvoice-cli")
        log_progress("Starting OpenVoice conversion (chunked to reduce VRAM)...", 0.0)
        try:
            out_path = openvoice_convert(
                input_path=input_path,
                ref_path=ref_path,
                output_path=None,
                device=None,
                tau=float(tau),
                temperature=float(temperature),
                chunk_duration_sec=18,
                progress_cb=log_progress,
                use_cpu_fallback_on_oom=True,
            )
        except Exception as e:
            raise gr.Error(f"OpenVoice conversion failed: {e}") from e
        log_progress("Done.", 1.0)
        return out_path


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
    topk, sync, tts_rate_pct, tts_pitch_hz,
    tts_backend, xtts_speed,
    chatterbox_exaggeration, chatterbox_cfg_weight,
    f5tts_ref_text, f5tts_model_path, f5tts_speed,
    input_audio_path,
    progress=gr.Progress(),
):
    """Tab 2 Step 2: TTS + optional voice conversion from edited segments."""
    if not edited_text or not edited_text.strip():
        raise gr.Error("No segments to synthesize. Run Step 1 first.")
    if total_duration == 0:
        raise gr.Error("Missing audio duration. Run Step 1 first.")

    ref_path = _audio_path(reference_audio) if reference_audio else None

    # Backends that need a reference voice — auto-fallback to input audio
    needs_ref = tts_backend in ("XTTS v2", "Chatterbox", "F5-TTS")
    if needs_ref and (not ref_path or not Path(ref_path).exists()):
        ref_path = input_audio_path if input_audio_path and Path(str(input_audio_path)).exists() else None
        if not ref_path:
            raise gr.Error(
                f"{tts_backend} requires a reference voice audio. "
                "Upload input audio in Step 1, or upload a separate Reference Voice Sample."
            )

    step_count = [0]

    def log(msg: str):
        step_count[0] += 1
        progress(min(0.1 + step_count[0] * 0.07, 0.92), desc=msg)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out_path = tmp.name

    if tts_backend == "XTTS v2":
        progress(0.05, desc="Loading XTTS v2 model (first run: downloads ~1.8 GB)...")
        tr.synthesize_from_edited(
            edited_text=edited_text,
            total_duration=float(total_duration),
            target_language=target_language,
            output_path=out_path,
            sync=bool(sync),
            tts_backend="xtts",
            xtts_ref_path=ref_path,
            xtts_speed=float(xtts_speed),
            progress_cb=log,
        )

    elif tts_backend == "Chatterbox":
        progress(0.05, desc="Loading Chatterbox model (first run: downloads ~1.5 GB)...")
        tr.synthesize_from_edited(
            edited_text=edited_text,
            total_duration=float(total_duration),
            target_language=target_language,
            output_path=out_path,
            sync=bool(sync),
            tts_backend="chatterbox",
            xtts_ref_path=ref_path,
            chatterbox_exaggeration=float(chatterbox_exaggeration),
            chatterbox_cfg_weight=float(chatterbox_cfg_weight),
            progress_cb=log,
        )

    elif tts_backend == "F5-TTS":
        model_path = f5tts_model_path.strip() if f5tts_model_path else None
        if model_path == "":
            model_path = None
        progress(0.05, desc="Loading F5-TTS model...")
        tr.synthesize_from_edited(
            edited_text=edited_text,
            total_duration=float(total_duration),
            target_language=target_language,
            output_path=out_path,
            sync=bool(sync),
            tts_backend="f5tts",
            xtts_ref_path=ref_path,
            f5tts_ref_text=f5tts_ref_text or "",
            f5tts_model_path=model_path,
            f5tts_speed=float(f5tts_speed),
            progress_cb=log,
        )

    else:
        # Edge-TTS (with optional OpenVoice)
        rate_str = f"{int(tts_rate_pct):+d}%"
        pitch_str = f"{int(tts_pitch_hz):+d}Hz"
        use_openvoice = tts_backend == "Edge-TTS + OpenVoice" and ref_path and Path(ref_path).exists()

        if use_openvoice:
            import tempfile as _tf
            with _tf.NamedTemporaryFile(suffix=".wav", delete=False) as _tmp:
                tts_only_path = _tmp.name
            tr.synthesize_from_edited(
                edited_text=edited_text,
                total_duration=float(total_duration),
                target_language=target_language,
                output_path=tts_only_path,
                sync=bool(sync),
                tts_rate=rate_str,
                tts_pitch=pitch_str,
                tts_backend="edge",
                progress_cb=log,
            )
            progress(0.85, desc="Applying OpenVoice tone-color transfer...")
            openvoice_convert(
                input_path=tts_only_path,
                ref_path=ref_path,
                output_path=out_path,
                progress_cb=lambda msg, _p=None: log(msg),
            )
            Path(tts_only_path).unlink(missing_ok=True)
        else:
            tr.synthesize_from_edited(
                edited_text=edited_text,
                total_duration=float(total_duration),
                target_language=target_language,
                output_path=out_path,
                sync=bool(sync),
                tts_rate=rate_str,
                tts_pitch=pitch_str,
                tts_backend="edge",
                progress_cb=log,
            )

    out_wav, _ = librosa.load(out_path, sr=SAMPLE_RATE, mono=True)
    Path(out_path).unlink(missing_ok=True)

    progress(1.0, desc="Done.")
    return (SAMPLE_RATE, out_wav)


def build_ui():
    with gr.Blocks(title="Voice Changer") as demo:
        gr.Markdown("# Voice Changer — Zero-Shot")

        with gr.Tabs():
            # ── Tab 1: Voice Conversion ──────────────────────────────────────
            with gr.Tab("Voice Conversion"):
                gr.Markdown(
                    "Skonwertuj głos z nagrania wejściowego na głos z próbki referencyjnej. "
                    "Trzy tryby: **Chatterbox VC** (resynteza przez S3), "
                    "**Chatterbox TTS** (transkrypcja → TTS z klonowaniem głosu), "
                    "**OpenVoice** (transfer tonu)."
                )
                with gr.Row():
                    with gr.Column():
                        vc_input = gr.Audio(label="Input Audio", type="filepath")
                        vc_ref = gr.Audio(
                            label="Reference Voice Sample (5–30 s)", type="filepath"
                        )
                        vc_model_radio = gr.Radio(
                            choices=["Chatterbox VC", "Chatterbox TTS", "OpenVoice"],
                            value="Chatterbox VC",
                            label="Conversion Model",
                        )
                        with gr.Group(visible=True) as vc_group_chatterbox_vc:
                            gr.Markdown(
                                "**Chatterbox VC** — resyntezuje audio zachowując treść, "
                                "aplikując tembr głosu z próbki referencyjnej (tokenizer S3). "
                                "Brak dodatkowych parametrów."
                            )
                        with gr.Group(visible=False) as vc_group_chatterbox_tts:
                            gr.Markdown("**Chatterbox TTS** — transkrypcja Whisper → synteza TTS z klonowaniem głosu")
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
                        with gr.Group(visible=False) as vc_group_openvoice:
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
                        vc_btn = gr.Button("Convert Voice", variant="primary")
                    with gr.Column():
                        vc_output = gr.Audio(label="Converted Audio", type="filepath")

                def _update_vc_model_ui(model):
                    return (
                        gr.update(visible=model == "Chatterbox VC"),
                        gr.update(visible=model == "Chatterbox TTS"),
                        gr.update(visible=model == "OpenVoice"),
                    )

                vc_model_radio.change(
                    fn=_update_vc_model_ui,
                    inputs=[vc_model_radio],
                    outputs=[vc_group_chatterbox_vc, vc_group_chatterbox_tts, vc_group_openvoice],
                )
                vc_btn.click(
                    fn=run_conversion,
                    inputs=[
                        vc_input, vc_ref, vc_model_radio,
                        vc_cb_exaggeration, vc_cb_cfg,
                        vc_tau, vc_temperature,
                    ],
                    outputs=[vc_output],
                )
                gr.Markdown(
                    "**Tips:** Use 5–30 s of clean reference speech. "
                    "Chatterbox VC/TTS download weights on first use. "
                    "Chatterbox TTS transcribes the input audio first, then re-synthesizes in the reference voice. "
                    "OpenVoice requires: `pip install openvoice-cli`."
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
                            value="Chatterbox",
                            label="TTS Engine",
                            info=(
                                "Chatterbox: najlepsza jakość, natywny polski, MIT. "
                                "F5-TTS: flow-matching, fine-tunable, wspólnotowy model PL. "
                                "XTTS v2: zero-shot klonowanie głosu. "
                                "Edge-TTS + OpenVoice: lekka opcja online."
                            ),
                        )
                        tr_ref = gr.Audio(
                            label="Reference Voice Sample — głos do sklonowania (10–30 s)",
                            type="filepath",
                        )
                        tr_sync = gr.Checkbox(
                            value=True,
                            label="Synchronize with source timing",
                            info="Adjust silence gaps to match source segment duration (speech is never stretched)",
                        )
                        with gr.Group(visible=True) as tr_group_chatterbox:
                            gr.Markdown("**Chatterbox settings**")
                            tr_cb_exaggeration = gr.Slider(
                                minimum=0.0, maximum=1.0, value=0.5, step=0.05,
                                label="Emotion exaggeration",
                                info="Higher = more expressive/dramatic speech.",
                            )
                            tr_cb_cfg = gr.Slider(
                                minimum=0.0, maximum=1.0, value=0.5, step=0.05,
                                label="CFG weight (guidance)",
                                info="Higher = more faithful to reference voice.",
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
                                placeholder='Puste = model bazowy. Wpisz "polish" dla modelu PL lub ścieżkę do .pt',
                                info='Use "polish" to auto-download the Polish community checkpoint (~3 GB).',
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
                    ],
                    outputs=[tr_output],
                )
                gr.Markdown(
                    "**Tips:** "
                    "**Chatterbox** needs 10–30 s of reference audio, downloads ~1.5 GB on first use. "
                    "**F5-TTS** needs 5–15 s reference + optional transcript; use `polish` model path for Polish. "
                    "**Fine-tuning F5-TTS:** `python finetune_f5tts.py --data_dir ./voice_data --base polish`"
                )



    return demo


def main():
    parser = argparse.ArgumentParser(description="Voice Conversion Web UI")
    parser.add_argument("--port", type=int, default=7862, help="Port to listen on")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument(
        "--low-vram", action="store_true",
        help="Reduce GPU chunk sizes to avoid OOM on cards with <8 GB VRAM "
             "(feature chunks: 5s, vocoder chunks: 300 frames)",
    )
    args = parser.parse_args()

    if args.low_vram:
        import voice_utils
        voice_utils.CHUNK_SECONDS = 5
        voice_utils.MATCH_CHUNK_FRAMES = 300
        print("Low-VRAM mode: feature chunks=5s, vocoder chunks=300 frames (~6s)")

    # kNN-VC is loaded only when using "Translate & Convert" with reference (saves ~9.5 GB VRAM for OpenVoice in Tab 1)

    demo = build_ui()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share, inbrowser=True)


if __name__ == "__main__":
    main()
