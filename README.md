# Voice Changer — Zero-Shot Voice Conversion

Speech-to-speech voice conversion without training.

- **Branch `natural-clone`:** **OpenVoice** tone-color transfer (more natural cloning, GPU).
- **Branch `gpu` / `main`:** **kNN-VC** (HuBERT/WavLM + k-NN + HiFiGAN).

## Branch: natural-clone (OpenVoice)

The **Voice Conversion** tab uses [OpenVoice](https://github.com/myshell-ai/OpenVoice) (via `openvoice-cli`) for natural voice cloning from a single reference sample.

### Installation (natural-clone)

```bash
git checkout natural-clone
pip install -r requirements.txt
```

For GPU (recommended):

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

If dependency conflicts appear (e.g. `librosa`), use a dedicated venv:

```bash
python -m venv .venv-openvoice
source .venv-openvoice/bin/activate  # or .venv-openvoice\Scripts\activate on Windows
pip install -r requirements.txt
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Usage (natural-clone)

- **Voice Conversion:** Upload input audio + reference voice → Convert. No Top-K or extra options; OpenVoice handles tone transfer.
- **Translate & Convert:** Unchanged (Whisper + TTS + optional kNN-VC).

---

## Branch: gpu / main (kNN-VC)

Based on [bshall/knn-vc](https://github.com/bshall/knn-vc) — HuBERT/WavLM features and k-NN matching.

## Requirements

- Python 3.9+
- PyTorch (CPU or CUDA)
- ~1.5 GB disk space for kNN-VC weights (downloaded automatically on first run)
- For natural-clone: `openvoice-cli` (see above)

## Installation (default)

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

## Docker

Pre-built `Dockerfile` + `docker-compose.yml` are included.

Requirements on the host:

- Docker Engine 25+ with Compose v2 (`docker compose version`)
- For GPU: the NVIDIA driver **and** the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (`nvidia-ctk`). The image ships its own CUDA libraries — only the driver has to be present on the host.

### GPU (default)

```bash
cp .env.example .env      # optional: set VC_PORT / HOST_AUDIO / ANTHROPIC_API_KEY
docker compose up --build
# open http://localhost:7862
```

Model downloads (Hugging Face, torch.hub, SpeechBrain, Argos Translate) are kept in named volumes, so they persist across `docker compose down`.

### CPU only

```bash
docker compose --profile cpu up --build app-cpu
```

### Custom input directory

By default `./data` on the host is mounted as `/data` in the container (read-write, so `ref_preprocess` can write its sidecar cache). Point the Gradio "input audio" / "reference" pickers there, or override via `HOST_AUDIO` in `.env`:

```bash
HOST_AUDIO=/media/wilk/PV1/pv/AUDIO docker compose up
```

### Useful overrides

```bash
# disable --low-vram:
docker compose run --rm app python app.py --host 0.0.0.0 --port 7862

# public Gradio tunnel:
docker compose run --rm -p 7862:7862 app python app.py --host 0.0.0.0 --port 7862 --share
```

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
