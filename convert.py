#!/usr/bin/env python3
"""
Zero-shot voice conversion using kNN-VC.
No training required - just provide input audio and a reference voice sample.

Usage:
    python convert.py --input input.wav --reference reference.wav --output output.wav
    python convert.py --input input.wav --reference reference.wav --output output.wav --topk 4
"""

import argparse
import sys
import torch
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path


SAMPLE_RATE = 16000
CHUNK_SECONDS = 15  # WavLM attention is O(N²); 15s chunks keep it manageable on CPU


def load_audio(path: str, target_sr: int = SAMPLE_RATE) -> torch.Tensor:
    """Load audio file and resample to target sample rate. Returns (1, T) tensor."""
    wav, _ = librosa.load(path, sr=target_sr, mono=True)
    return torch.from_numpy(wav).unsqueeze(0)  # (1, T)


def extract_features_chunked(knn_vc, wav: torch.Tensor, chunk_seconds: int = CHUNK_SECONDS) -> torch.Tensor:
    """Extract WavLM features in chunks to avoid O(N²) attention hang on long audio."""
    chunk_samples = chunk_seconds * SAMPLE_RATE
    total_samples = wav.shape[1]

    if total_samples <= chunk_samples:
        return knn_vc.get_features(wav)

    chunks = []
    n_chunks = (total_samples + chunk_samples - 1) // chunk_samples
    for i, start in enumerate(range(0, total_samples, chunk_samples)):
        end = min(start + chunk_samples, total_samples)
        feats = knn_vc.get_features(wav[:, start:end])
        chunks.append(feats)
        print(f"  Features: chunk {i+1}/{n_chunks} ({end/SAMPLE_RATE:.0f}s / {total_samples/SAMPLE_RATE:.0f}s)")

    return torch.cat(chunks, dim=0)


def convert_voice(
    input_path: str,
    reference_path: str,
    output_path: str,
    topk: int = 4,
    device: str = "auto",
) -> None:
    """
    Convert voice in input_path to match the voice in reference_path.

    Args:
        input_path: Path to the source audio file (voice to convert).
        reference_path: Path to the reference audio file (target voice sample).
        output_path: Path to save the converted audio.
        topk: Number of nearest neighbors to use. Higher = smoother but less expressive.
        device: 'cuda', 'cpu', or 'auto'.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"Loading kNN-VC model from torch.hub (bshall/knn-vc)...")
    print("  (First run will download ~1.5 GB of model weights)")

    knn_vc = torch.hub.load(
        "bshall/knn-vc",
        "knn_vc",
        prematched=True,
        trust_repo=True,
        device=device,
    )
    knn_vc.eval()

    print(f"Loading input audio:     {input_path}")
    src_wav = load_audio(input_path).to(device)

    print(f"Loading reference audio: {reference_path}")
    ref_wav = load_audio(reference_path).to(device)

    print("Extracting features from input audio...")
    with torch.inference_mode():
        query_seq = extract_features_chunked(knn_vc, src_wav)

    print("Building matching set from reference audio...")
    with torch.inference_mode():
        matching_set = knn_vc.get_matching_set([ref_wav])

    print(f"Running kNN voice conversion (topk={topk})...")
    with torch.inference_mode():
        out_wav = knn_vc.match(query_seq, matching_set, topk=topk)

    out_wav = out_wav.squeeze().cpu().numpy()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), out_wav, SAMPLE_RATE)

    duration = len(out_wav) / SAMPLE_RATE
    print(f"Saved converted audio ({duration:.1f}s) to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot voice conversion using kNN-VC (no training required)"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input audio file (voice to convert)",
    )
    parser.add_argument(
        "--reference", "-r",
        required=True,
        help="Reference audio file (target voice sample, 5-30s recommended)",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output audio file path",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=4,
        help="Number of nearest neighbors (default: 4). Higher = smoother output.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Device to run on (default: auto)",
    )

    args = parser.parse_args()

    for path, name in [(args.input, "input"), (args.reference, "reference")]:
        if not Path(path).exists():
            print(f"Error: {name} file not found: {path}", file=sys.stderr)
            sys.exit(1)

    convert_voice(
        input_path=args.input,
        reference_path=args.reference,
        output_path=args.output,
        topk=args.topk,
        device=args.device,
    )


if __name__ == "__main__":
    main()
