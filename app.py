#!/usr/bin/env python3
"""
Gradio web interface for zero-shot voice conversion using kNN-VC.

Usage:
    python app.py
    python app.py --share   # public URL via Gradio tunnel
    python app.py --port 7861
"""

import argparse
import tempfile
from pathlib import Path

import gradio as gr
import librosa
import numpy as np
import soundfile as sf
import torch

import translate as tr
from voice_utils import extract_features_chunked

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

    with torch.inference_mode():
        query_seq = extract_features_chunked(
            knn_vc, src_wav,
            progress_cb=lambda msg: progress(None, desc=msg),
        )
        progress(0.70, desc="Building reference matching set...")
        matching_set = knn_vc.get_matching_set([ref_wav])
        progress(0.80, desc="Running kNN matching + vocoding...")
        out_wav = knn_vc.match(query_seq, matching_set, topk=int(topk))

    progress(1.0, desc="Done.")
    return (SAMPLE_RATE, out_wav.squeeze().cpu().numpy())


def run_translate_convert(
    input_audio, reference_audio, target_language, topk, sync,
    progress=gr.Progress(),
):
    """Tab 2: Translate speech and optionally apply voice conversion."""
    if input_audio is None:
        raise gr.Error("Please upload an input audio file.")

    knn_vc = get_model()
    device = next(knn_vc.parameters()).device

    matching_set = None
    if reference_audio is not None:
        progress(0.05, desc="Building reference voice matching set...")
        ref_wav = load_audio_array(reference_audio).to(device)
        with torch.inference_mode():
            matching_set = knn_vc.get_matching_set([ref_wav])

    step_count = [0]

    def log(msg: str):
        step_count[0] += 1
        progress(min(0.1 + step_count[0] * 0.10, 0.90), desc=msg)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        out_path = tmp.name

    english_text, translated_text = tr.run_pipeline(
        audio_path=input_audio,
        target_language=target_language,
        output_path=out_path,
        knn_vc=knn_vc if matching_set is not None else None,
        matching_set=matching_set,
        topk=int(topk),
        sync=bool(sync),
        progress_cb=log,
    )

    out_wav, _ = librosa.load(out_path, sr=SAMPLE_RATE, mono=True)
    Path(out_path).unlink(missing_ok=True)

    progress(1.0, desc="Done.")
    transcript = f"[English]\n{english_text}\n\n[{target_language}]\n{translated_text}"
    return (SAMPLE_RATE, out_wav), transcript


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
                    "Pipeline: **Whisper ASR** → **argostranslate** → "
                    "**edge-tts** → *(optional)* **kNN-VC voice conversion**"
                )
                with gr.Row():
                    with gr.Column():
                        tr_input = gr.Audio(label="Input Audio (any language)", type="filepath")
                        tr_ref = gr.Audio(
                            label="Reference Voice Sample (optional, for voice conversion)",
                            type="filepath",
                        )
                        tr_lang = gr.Dropdown(
                            choices=list(tr.LANGUAGES.keys()),
                            value="Polish",
                            label="Target Language",
                        )
                        tr_sync = gr.Checkbox(
                            value=True,
                            label="Synchronize with source timing",
                            info="Each segment is time-stretched to match the original audio timeline",
                        )
                        tr_topk = gr.Slider(
                            minimum=1, maximum=16, value=4, step=1,
                            label="Top-K Neighbors (used only with reference voice)",
                            info="Higher = smoother, lower = more expressive",
                        )
                        tr_btn = gr.Button("Translate & Convert", variant="primary")
                    with gr.Column():
                        tr_output = gr.Audio(label="Output Audio", type="numpy")
                        tr_transcript = gr.Textbox(
                            label="Transcription & Translation",
                            lines=6,
                            interactive=False,
                        )

                tr_btn.click(
                    fn=run_translate_convert,
                    inputs=[tr_input, tr_ref, tr_lang, tr_topk, tr_sync],
                    outputs=[tr_output, tr_transcript],
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
    args = parser.parse_args()

    get_model()

    demo = build_ui()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share, inbrowser=True)


if __name__ == "__main__":
    main()
