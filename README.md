# Voice Changer — Zero-Shot Voice Conversion

Speech-to-speech voice conversion without training. Based on **kNN-VC**
([bshall/knn-vc](https://github.com/bshall/knn-vc)) — uses HuBERT/WavLM features
and k-nearest neighbor matching to transfer a speaker's voice style.

## Requirements

- Python 3.9+
- PyTorch (CPU or CUDA)
- ~1.5 GB disk space for model weights (downloaded automatically on first run)

## Installation

```bash
pip install -r requirements.txt
```

For GPU support, install PyTorch with CUDA:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Usage

### CLI

```bash
python convert.py --input input.wav --reference reference.wav --output output.wav
```

Options:
- `--input / -i`     — source audio (voice to convert)
- `--reference / -r` — target voice sample (5-30s recommended)
- `--output / -o`    — output file path
- `--topk`           — number of nearest neighbors (default: 4, higher = smoother)
- `--device`         — `auto` | `cuda` | `cpu`

### Web UI

```bash
python app.py
# then open http://localhost:7860
```

Options:
- `--port 7861`  — custom port
- `--share`      — create public Gradio tunnel URL
- `--host 0.0.0.0` — listen on all interfaces

## How it works

1. HuBERT extracts frame-level speech features from the **input** audio
2. WavLM extracts features from the **reference** audio, building a matching set
3. For each input frame, the k nearest neighbors from the reference set are found
4. HiFiGAN vocoder reconstructs the audio using the matched reference features

Result: the linguistic content of the input, but in the voice of the reference speaker.

## Tips for best results

- Reference sample: clean speech, no background music/noise
- Reference duration: 15-30 seconds works best
- Supported formats: WAV, MP3, FLAC, OGG
- Both audios are automatically resampled to 16 kHz
