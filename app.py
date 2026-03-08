#!/usr/bin/env python3
"""
Gradio web interface for zero-shot voice conversion using kNN-VC.

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
import numpy as np
import soundfile as sf
import torch

import translate as tr
from voice_utils import extract_features_chunked, get_matching_set_chunked, match_chunked

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


def load_audio_array(path: str) -> torch.Tensor:
    wav, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return torch.from_numpy(wav).unsqueeze(0)  # (1, T)


def run_conversion(input_audio, reference_audio, topk: int, progress=gr.Progress()):
    """Tab 1: Voice conversion without translation."""
    if input_audio is None:
        raise gr.Error("Please upload an input audio file.")
    if reference_audio is None:
        raise gr.Error("Please upload a reference voice sample.")

    knn_vc = get_model()
    device = next(knn_vc.parameters()).device

    progress(0.0, desc="Loading audio...")
    src_wav = load_audio_array(input_audio).to(device)
    ref_wav = load_audio_array(reference_audio).to(device)

    def log(msg: str):
        print(msg)
        progress(None, desc=msg)

    with torch.inference_mode():
        # Build matching set FIRST while VRAM is clean (chunked — handles long ref audio)
        progress(0.10, desc="Building reference matching set...")
        matching_set = get_matching_set_chunked(knn_vc, ref_wav)
        torch.cuda.empty_cache()

        # Extract source features (chunked; periodic cache flush inside)
        query_seq = extract_features_chunked(knn_vc, src_wav, progress_cb=log)
        torch.cuda.empty_cache()

        n_frames = query_seq.shape[0]
        progress(0.80, desc=f"Vocoding {n_frames} frames in chunks...")
        out_wav = match_chunked(knn_vc, query_seq, matching_set, topk=int(topk), progress_cb=log)

    torch.cuda.empty_cache()
    progress(1.0, desc="Done.")
    return (SAMPLE_RATE, out_wav.squeeze().cpu().numpy())


def run_step1_transcribe(
    input_audio, target_language, female_narrator,
    progress=gr.Progress(),
):
    """Tab 2 Step 1: Whisper ASR → translate → return editable segment text."""
    if input_audio is None:
        raise gr.Error("Please upload an input audio file.")

    step_count = [0]

    def log(msg: str):
        step_count[0] += 1
        progress(min(0.05 + step_count[0] * 0.08, 0.95), desc=msg)

    segments_text, total_duration = tr.transcribe_and_translate(
        audio_path=input_audio,
        target_language=target_language,
        female_narrator=bool(female_narrator),
        progress_cb=log,
    )
    progress(1.0, desc="Done. Edit segments below, then click Synthesize.")
    return segments_text, total_duration


def run_step2_synthesize(
    edited_text, total_duration, reference_audio, target_language,
    topk, sync, tts_rate_pct, tts_pitch_hz,
    progress=gr.Progress(),
):
    """Tab 2 Step 2: TTS + optional voice conversion from edited segments."""
    if not edited_text or not edited_text.strip():
        raise gr.Error("No segments to synthesize. Run Step 1 first.")
    if total_duration == 0:
        raise gr.Error("Missing audio duration. Run Step 1 first.")

    rate_str = f"{int(tts_rate_pct):+d}%"
    pitch_str = f"{int(tts_pitch_hz):+d}Hz"

    knn_vc = get_model()
    device = next(knn_vc.parameters()).device

    matching_set = None
    if reference_audio is not None:
        progress(0.05, desc="Building reference voice matching set...")
        ref_wav = load_audio_array(reference_audio).to(device)
        with torch.inference_mode():
            matching_set = get_matching_set_chunked(knn_vc, ref_wav)

    step_count = [0]

    def log(msg: str):
        step_count[0] += 1
        progress(min(0.1 + step_count[0] * 0.10, 0.90), desc=msg)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out_path = tmp.name

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
                    "Convert voice in input audio to match a reference speaker. "
                    "No training required."
                )
                with gr.Row():
                    with gr.Column():
                        vc_input = gr.Audio(label="Input Audio", type="filepath")
                        vc_ref = gr.Audio(
                            label="Reference Voice Sample (5-30s)", type="filepath"
                        )
                        vc_topk = gr.Slider(
                            minimum=1, maximum=16, value=4, step=1,
                            label="Top-K Neighbors",
                            info="Higher = smoother, lower = more expressive",
                        )
                        vc_btn = gr.Button("Convert Voice", variant="primary")
                    with gr.Column():
                        vc_output = gr.Audio(label="Converted Audio", type="numpy")

                vc_btn.click(
                    fn=run_conversion,
                    inputs=[vc_input, vc_ref, vc_topk],
                    outputs=[vc_output],
                )
                gr.Markdown("**Tips:** Reference should be clean speech, 15-30s. Supports WAV, MP3, FLAC, OGG.")

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
                    # ── Step 2 inputs ────────────────────────────────────────
                    with gr.Column():
                        gr.Markdown("### Step 2 — Synthesize & Convert")
                        tr_ref = gr.Audio(
                            label="Reference Voice Sample (optional, for voice conversion)",
                            type="filepath",
                        )
                        tr_sync = gr.Checkbox(
                            value=True,
                            label="Synchronize with source timing",
                            info="Adjust silence gaps to match source segment duration (speech is never stretched)",
                        )
                        tr_topk = gr.Slider(
                            minimum=1, maximum=16, value=4, step=1,
                            label="Top-K Neighbors (used only with reference voice)",
                            info="Higher = smoother, lower = more expressive",
                        )
                        tr_rate = gr.Slider(
                            minimum=-30, maximum=30, value=0, step=1,
                            label="TTS Speaking Rate (%)",
                            info="Negative = slower, positive = faster. Default: 0",
                        )
                        tr_pitch = gr.Slider(
                            minimum=-20, maximum=20, value=0, step=1,
                            label="TTS Pitch (Hz)",
                            info="Negative = deeper voice, positive = higher pitch. Default: 0",
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
                tr_step2_btn.click(
                    fn=run_step2_synthesize,
                    inputs=[tr_segments, tr_duration, tr_ref, tr_lang, tr_topk, tr_sync, tr_rate, tr_pitch],
                    outputs=[tr_output],
                )
                gr.Markdown(
                    "**Tips:** "
                    "Reference voice is optional — without it you get translated TTS only. "
                    "First use downloads Whisper (~150 MB) and translation package (~80 MB per language)."
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

    get_model()

    demo = build_ui()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share, inbrowser=True)


if __name__ == "__main__":
    main()
