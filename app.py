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

SAMPLE_RATE = 16000
CHUNK_SECONDS = 15  # WavLM attention is O(N²); 15s chunks keep it manageable on CPU
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


def extract_features_chunked(knn_vc, wav: torch.Tensor, progress: gr.Progress, base: float, span: float) -> torch.Tensor:
    """Extract WavLM features in chunks to avoid O(N²) attention hang on long audio."""
    chunk_samples = CHUNK_SECONDS * SAMPLE_RATE
    total_samples = wav.shape[1]

    if total_samples <= chunk_samples:
        progress(base + span, desc="Extracting features...")
        return knn_vc.get_features(wav)

    chunks = []
    n_chunks = (total_samples + chunk_samples - 1) // chunk_samples
    for i, start in enumerate(range(0, total_samples, chunk_samples)):
        end = min(start + chunk_samples, total_samples)
        feats = knn_vc.get_features(wav[:, start:end])
        chunks.append(feats)
        frac = (i + 1) / n_chunks
        progress(base + span * frac, desc=f"Extracting features: {end//SAMPLE_RATE}s / {total_samples//SAMPLE_RATE}s")

    return torch.cat(chunks, dim=0)


def run_conversion(input_audio, reference_audio, topk: int, progress=gr.Progress()):
    """Gradio handler: takes file paths, returns (sample_rate, numpy_array)."""
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
        # Features for input: chunked (can be long)
        query_seq = extract_features_chunked(knn_vc, src_wav, progress, base=0.05, span=0.60)

        # Features for reference (usually short, one chunk)
        progress(0.70, desc="Building reference matching set...")
        matching_set = knn_vc.get_matching_set([ref_wav])

        progress(0.80, desc="Running kNN matching + vocoding...")
        out_wav = knn_vc.match(query_seq, matching_set, topk=int(topk))

    progress(1.0, desc="Done.")
    out_np = out_wav.squeeze().cpu().numpy()
    return (SAMPLE_RATE, out_np)


def build_ui():
    with gr.Blocks(title="Zero-Shot Voice Conversion") as demo:
        gr.Markdown(
            """
            # Zero-Shot Voice Conversion
            Convert any voice to match a reference speaker — no training required.

            **How it works:** Upload the audio you want to convert and a short sample
            of the target voice (5-30 seconds). The model uses k-nearest neighbor
            matching on HuBERT/WavLM features to transfer the voice style.
            """
        )

        with gr.Row():
            with gr.Column():
                input_audio = gr.Audio(
                    label="Input Audio (voice to convert)",
                    type="filepath",
                )
                reference_audio = gr.Audio(
                    label="Reference Voice Sample (target speaker, 5-30s)",
                    type="filepath",
                )
                topk = gr.Slider(
                    minimum=1,
                    maximum=16,
                    value=4,
                    step=1,
                    label="Top-K Neighbors",
                    info="Higher = smoother output, lower = more expressive",
                )
                convert_btn = gr.Button("Convert Voice", variant="primary")

            with gr.Column():
                output_audio = gr.Audio(
                    label="Converted Audio",
                    type="numpy",
                )

        convert_btn.click(
            fn=run_conversion,
            inputs=[input_audio, reference_audio, topk],
            outputs=[output_audio],
        )

        gr.Markdown(
            """
            **Tips:**
            - Reference sample should be clean speech (no background music/noise)
            - Longer reference audio (15-30s) generally gives better results
            - Input and reference should be WAV, MP3, FLAC, or OGG
            - First run downloads ~1.5 GB of model weights (cached for future runs)
            """
        )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Voice Conversion Web UI")
    parser.add_argument("--port", type=int, default=7862, help="Port to listen on")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    args = parser.parse_args()

    # Pre-load model before starting UI
    get_model()

    demo = build_ui()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share, inbrowser=True)


if __name__ == "__main__":
    main()
