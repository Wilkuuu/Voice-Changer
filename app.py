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
from voice_utils import get_matching_set_chunked

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


def run_conversion(input_audio, reference_audio, tau, temperature, progress=gr.Progress()):
    """Tab 1: Voice conversion using OpenVoice (natural tone-color transfer)."""
    input_path = _audio_path(input_audio)
    ref_path = _audio_path(reference_audio)
    if not input_path or not Path(input_path).exists():
        raise gr.Error("Please upload an input audio file.")
    if not ref_path or not Path(ref_path).exists():
        raise gr.Error("Please upload a reference voice sample.")

    if not openvoice_available():
        raise gr.Error(
            "OpenVoice is not installed. Install it for natural voice cloning:\n"
            "  pip install openvoice-cli\n"
            "For GPU: pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118"
        )

    def log_progress(msg: str, p: float | None = None) -> None:
        print(f"[Voice Conversion] {msg}", flush=True)
        if p is not None:
            progress(float(p), desc=msg)
        else:
            progress(None, desc=msg)

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
        print(f"[Voice Conversion] ERROR: {e}", flush=True)
        raise gr.Error(f"Voice conversion failed: {e}") from e
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
    return segments_text, total_duration


def run_step2_synthesize(
    edited_text, total_duration, reference_audio, target_language,
    topk, sync, tts_rate_pct, tts_pitch_hz,
    tts_backend, xtts_speed,
    progress=gr.Progress(),
):
    """Tab 2 Step 2: TTS + optional voice conversion from edited segments."""
    if not edited_text or not edited_text.strip():
        raise gr.Error("No segments to synthesize. Run Step 1 first.")
    if total_duration == 0:
        raise gr.Error("Missing audio duration. Run Step 1 first.")

    ref_path = _audio_path(reference_audio) if reference_audio else None
    use_xtts = tts_backend == "XTTS v2"

    if use_xtts and (not ref_path or not Path(ref_path).exists()):
        raise gr.Error("XTTS v2 requires a Reference Voice Sample for voice cloning.")

    step_count = [0]

    def log(msg: str):
        step_count[0] += 1
        progress(min(0.1 + step_count[0] * 0.07, 0.92), desc=msg)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out_path = tmp.name

    if use_xtts:
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
    else:
        rate_str = f"{int(tts_rate_pct):+d}%"
        pitch_str = f"{int(tts_pitch_hz):+d}Hz"

        knn_vc = get_model()
        device = next(knn_vc.parameters()).device

        matching_set = None
        if ref_path and Path(ref_path).exists():
            progress(0.05, desc="Building reference voice matching set...")
            ref_wav = load_audio_array(ref_path).to(device)
            with torch.inference_mode():
                matching_set = get_matching_set_chunked(knn_vc, ref_wav)

        tr.synthesize_from_edited(
            edited_text=edited_text,
            total_duration=float(total_duration),
            target_language=target_language,
            output_path=out_path,
            knn_vc=knn_vc if matching_set is not None else None,
            matching_set=matching_set,
            topk=int(topk),
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
                    "Convert voice in input audio to match a reference speaker using **OpenVoice** "
                    "(tone-color transfer). Natural cloning, no training, GPU-accelerated."
                )
                with gr.Row():
                    with gr.Column():
                        vc_input = gr.Audio(label="Input Audio", type="filepath")
                        vc_ref = gr.Audio(
                            label="Reference Voice Sample (5–30 s)", type="filepath"
                        )
                        vc_tau = gr.Slider(
                            minimum=0.3,
                            maximum=1.0,
                            value=0.65,
                            step=0.05,
                            label="Reference voice strength (tau)",
                            info="Higher = output closer to reference.",
                        )
                        vc_temperature = gr.Slider(
                            minimum=0.7,
                            maximum=1.3,
                            value=1.0,
                            step=0.05,
                            label="Voice temperature",
                            info="Higher = warmer, more expressive; lower = calmer, more even.",
                        )
                        vc_btn = gr.Button("Convert Voice", variant="primary")
                    with gr.Column():
                        vc_output = gr.Audio(label="Converted Audio", type="filepath")

                vc_btn.click(
                    fn=run_conversion,
                    inputs=[vc_input, vc_ref, vc_tau, vc_temperature],
                    outputs=[vc_output],
                )
                gr.Markdown(
                    "**Tips:** Use 5–30 s of clean reference speech (one speaker, little noise). "
                    "Requires: `pip install openvoice-cli` (and CUDA PyTorch for GPU)."
                )

            # ── Tab 2: Translate & Convert ───────────────────────────────────
            with gr.Tab("Translate & Convert"):
                gr.Markdown(
                    "Translate speech from **any language** to a target language, "
                    "then optionally apply voice conversion to match a reference speaker.\n\n"
                    "**Step 1:** Transcribe & translate → review/edit segments → "
                    "**Step 2:** Synthesize & convert"
                )
                # Shared state: total audio duration returned by step 1
                tr_duration = gr.State(0.0)

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
                            placeholder="e.g. Erotic monologue of a girl speaking to a male listener. Polish language.",
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
                            choices=["Edge-TTS + kNN-VC", "XTTS v2"],
                            value="XTTS v2",
                            label="TTS Engine",
                            info=(
                                "XTTS v2: natural voice cloning in one step — requires reference audio, "
                                "downloads ~1.8 GB on first use. "
                                "Edge-TTS + kNN-VC: faster, works without reference."
                            ),
                        )
                        tr_ref = gr.Audio(
                            label="Reference Voice Sample (3–30 s, required for XTTS / optional for kNN-VC)",
                            type="filepath",
                        )
                        tr_sync = gr.Checkbox(
                            value=True,
                            label="Synchronize with source timing",
                            info="Adjust silence gaps to match source segment duration (speech is never stretched)",
                        )
                        with gr.Group():
                            tr_xtts_speed = gr.Slider(
                                minimum=0.5, maximum=2.0, value=1.0, step=0.05,
                                label="XTTS Speaking Speed",
                                info="Only for XTTS v2. 1.0 = normal speed.",
                            )
                            tr_topk = gr.Slider(
                                minimum=1, maximum=16, value=4, step=1,
                                label="Top-K Neighbors (Edge-TTS + kNN-VC only)",
                                info="Higher = smoother, lower = more expressive",
                            )
                            tr_rate = gr.Slider(
                                minimum=-30, maximum=30, value=0, step=1,
                                label="TTS Speaking Rate % (Edge-TTS only)",
                                info="Negative = slower, positive = faster.",
                            )
                            tr_pitch = gr.Slider(
                                minimum=-20, maximum=20, value=0, step=1,
                                label="TTS Pitch Hz (Edge-TTS only)",
                                info="Negative = deeper, positive = higher pitch.",
                            )
                        tr_step2_btn = gr.Button("Synthesize & Convert", variant="primary")

                    # ── Output ───────────────────────────────────────────────
                    with gr.Column():
                        gr.Markdown("### Output")
                        tr_output = gr.Audio(label="Output Audio", type="numpy")

                tr_step1_btn.click(
                    fn=run_step1_transcribe,
                    inputs=[tr_input, tr_lang, tr_female],
                    outputs=[tr_segments, tr_duration],
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
                tr_step2_btn.click(
                    fn=run_step2_synthesize,
                    inputs=[
                        tr_segments, tr_duration, tr_ref, tr_lang,
                        tr_topk, tr_sync, tr_rate, tr_pitch,
                        tr_tts_backend, tr_xtts_speed,
                    ],
                    outputs=[tr_output],
                )
                gr.Markdown(
                    "**Tips:** "
                    "XTTS v2 needs 3–30 s of clean reference speech and ~4 GB VRAM (CPU fallback available, slow). "
                    "First use downloads Whisper (~150 MB), translation packages (~80 MB), and XTTS v2 (~1.8 GB)."
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
