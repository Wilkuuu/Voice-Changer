#!/usr/bin/env python3
"""
Fine-tune F5-TTS on a custom voice dataset.

This script prepares your audio data and launches the F5-TTS fine-tuning
pipeline. Fine-tuning creates a personalized TTS model that synthesizes speech
in the timbre of your target speaker.

Usage:
    python finetune_f5tts.py --data_dir ./my_voice_data --output_dir ./my_model
    python finetune_f5tts.py --data_dir ./my_voice_data --base polish --epochs 200

Data format (--data_dir):
    Each audio clip must have a matching .txt file with its transcript:
        clip_001.wav
        clip_001.txt   (contains: "Cześć, jak się masz?")
        clip_002.wav
        clip_002.txt
        ...

    Requirements:
    - Minimum: ~10 hours of audio for acceptable quality
    - Recommended: 20–90 hours for professional quality
    - Audio: 16–24 kHz, mono, clean (minimal background noise)
    - Single speaker per dataset

Base models:
    --base base      : F5TTS_v1_Base (English, best starting point for most languages)
    --base polish    : Gregniuki/F5-tts_English_German_Polish (better for Polish!)
                       Downloaded automatically from HuggingFace (~3 GB)

Hardware:
    - Minimum: RTX 3090 24 GB (slow)
    - Recommended: RTX 4090 24 GB or A100
    - GPU memory: ~18–22 GB for batch_size=4

After training:
    Use the checkpoint in the app's F5-TTS engine:
        Model path: ./my_model/checkpoint_final.pt
    Or via code:
        import f5tts_engine
        f5tts_engine.synthesize(text, ref_path, output, model_path="./my_model/checkpoint_final.pt")
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def prepare_metadata(data_dir: Path, output_dir: Path) -> Path:
    """
    Scan data_dir for wav+txt pairs, create metadata.csv for F5-TTS training.

    Returns path to generated metadata.csv.
    """
    clips = []
    for wav_path in sorted(data_dir.glob("*.wav")):
        txt_path = wav_path.with_suffix(".txt")
        if not txt_path.exists():
            print(f"  [skip] No transcript for {wav_path.name}")
            continue
        transcript = txt_path.read_text(encoding="utf-8").strip()
        if not transcript:
            continue
        clips.append((wav_path.resolve(), transcript))

    if not clips:
        print(f"ERROR: No valid wav+txt pairs found in {data_dir}")
        sys.exit(1)

    print(f"Found {len(clips)} clips")

    # Compute total duration
    total_sec = 0.0
    try:
        import librosa
        for wav_path, _ in clips:
            dur = librosa.get_duration(path=str(wav_path))
            total_sec += dur
        print(f"Total audio: {total_sec/3600:.2f} hours ({total_sec/60:.1f} minutes)")
        if total_sec < 3600:
            print("WARNING: Less than 1 hour of audio. Quality may be poor.")
        elif total_sec < 36000:
            print(f"INFO: {total_sec/3600:.1f}h — acceptable. 20-90h gives professional quality.")
    except ImportError:
        print("(Install librosa for duration check)")

    # Write metadata.csv in F5-TTS format: filename|transcript
    metadata_path = output_dir / "metadata.csv"
    with open(metadata_path, "w", encoding="utf-8") as f:
        for wav_path, transcript in clips:
            f.write(f"{wav_path}|{transcript}\n")

    print(f"Metadata written: {metadata_path}")
    return metadata_path


def download_base_checkpoint(base: str) -> str:
    """Download base model checkpoint. Returns path to .pt file."""
    if base == "base":
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id="SWivid/F5-TTS",
            filename="F5TTS_v1_Base/model_1200000.pt",
        )
        print(f"Base checkpoint: {path}")
        return path
    elif base == "polish":
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id="Gregniuki/F5-tts_English_German_Polish",
            filename="Polish/model_500000.pt",
        )
        print(f"Polish community checkpoint: {path}")
        return path
    else:
        # Treat as a local path
        if not Path(base).exists():
            print(f"ERROR: Base checkpoint not found: {base}")
            sys.exit(1)
        return base


def write_training_config(
    output_dir: Path,
    metadata_path: Path,
    base_ckpt: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> Path:
    """Write a minimal F5-TTS training config YAML."""
    config = {
        "model": "F5TTS",
        "dataset": {
            "name": "custom",
            "metadata_csv": str(metadata_path),
            "sample_rate": 24000,
        },
        "training": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "warmup_steps": 1000,
            "save_per_updates": 5000,
            "last_per_steps": 2000,
            "checkpoint": base_ckpt,
            "output_dir": str(output_dir),
        },
    }
    config_path = output_dir / "train_config.json"
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Training config: {config_path}")
    return config_path


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune F5-TTS on a custom voice dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data_dir", required=True, type=Path,
                        help="Directory with .wav + .txt pairs")
    parser.add_argument("--output_dir", default="./finetuned_model", type=Path,
                        help="Output directory for the fine-tuned model")
    parser.add_argument("--base", default="polish",
                        help="Base model: 'base', 'polish', or path to .pt file (default: polish)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs (default: 100; use 200+ for better quality)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size (reduce to 1-2 if OOM, default: 4)")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate (default: 1e-5)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Prepare data and config only, do not start training")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=== F5-TTS Fine-Tuning Setup ===")
    print(f"Data:        {args.data_dir}")
    print(f"Output:      {args.output_dir}")
    print(f"Base model:  {args.base}")
    print(f"Epochs:      {args.epochs}")
    print(f"Batch size:  {args.batch_size}")
    print()

    # Step 1: Prepare metadata
    print("Step 1: Scanning dataset...")
    metadata_path = prepare_metadata(args.data_dir, args.output_dir)

    # Step 2: Download base checkpoint
    print("\nStep 2: Downloading base checkpoint...")
    base_ckpt = download_base_checkpoint(args.base)

    # Step 3: Write config
    print("\nStep 3: Writing training config...")
    config_path = write_training_config(
        args.output_dir, metadata_path, base_ckpt,
        args.epochs, args.batch_size, args.lr,
    )

    if args.dry_run:
        print("\nDry run complete. To start training:")
        print(f"  f5-tts_train --config {config_path}")
        return

    # Step 4: Launch training
    print("\nStep 4: Starting F5-TTS fine-tuning...")
    print("This will take several hours. Press Ctrl+C to stop (checkpoint is saved periodically).")
    print()

    try:
        subprocess.run(
            [sys.executable, "-m", "f5_tts.train", "--config", str(config_path)],
            check=True,
        )
    except FileNotFoundError:
        # Try the CLI entrypoint
        subprocess.run(
            ["f5-tts_train", "--config", str(config_path)],
            check=True,
        )

    print(f"\nTraining complete. Model saved in: {args.output_dir}")
    print(f"\nTo use in the app, set F5-TTS Model Path to:")
    best_ckpt = max(
        args.output_dir.glob("*.pt"),
        key=lambda p: p.stat().st_mtime,
        default=None,
    )
    if best_ckpt:
        print(f"  {best_ckpt}")


if __name__ == "__main__":
    main()
