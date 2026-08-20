# Voice Changer — Zero-Shot Voice Conversion & TTS

Speech-to-speech voice conversion and voice-cloned TTS without per-speaker training.

## Features

- **Voice Conversion** — Chatterbox VC, Seed-VC, OpenVoice, Chatterbox TTS (ASR → resynthesis)
- **Translate & Convert** — Whisper/Argos translation or SRT import → cloned TTS (F5-TTS, Chatterbox, XTTS v2, Edge-TTS)
- **Tagged TTS** — plain-text scripts with optional Bark tags (`[śmiech]`, `[pauza]`, …)
- **Prepare Reference** — VAD, denoise, LUFS normalization, best-window selection

## Requirements

- Python 3.9+ (3.12 supported with `coqui-tts` fork)
- PyTorch (CPU or CUDA)
- GPU recommended for Chatterbox / F5-TTS / Seed-VC (~6–8 GB VRAM)

## Installation

```bash
pip install -r requirements.txt
```

For GPU support:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Optional engines (install as needed):

```bash
pip install chatterbox-tts    # Chatterbox TTS + VC
pip install f5-tts            # F5-TTS (recommended for Polish)
pip install coqui-tts         # XTTS v2
pip install openvoice-cli     # OpenVoice tone transfer
pip install seed-vc           # Seed-VC
```

## Usage

### Web UI (full app)

```bash
python app.py
# open http://localhost:7862
```

Options: `--port`, `--share`, `--host 0.0.0.0`, `--low-vram`, `--cpu`

### Simple VC UI

```bash
python simple_vc.py --port 7870
```

### CLI (kNN-VC)

```bash
python convert.py -i input.wav -r reference.wav -o output.wav --topk 4
```

## Docker

```bash
cp .env.example .env
docker compose up --build
# open http://localhost:7862
```

CPU-only: `docker compose --profile cpu up --build app-cpu`

Model caches persist in Docker volumes. Runtime artifacts (`processed/`, `checkpoints/`, `logs/`) are gitignored.

## TTS quality tips (Polish)

- Use **F5-TTS** with the Polish community checkpoint (`polish` shorthand) and a clean 10–30 s reference
- Provide or auto-generate a **reference transcript** for F5-TTS
- Keep **Strict timing sync** off unless dubbing to fixed SRT slots; use gentle sync for subtitle alignment without speech compression
- Enable **reference preprocessing** (VAD / denoise / best window)
- Output is **24 kHz** for clone-TTS backends (Chatterbox, F5-TTS, XTTS)

## Tips for voice conversion

- Reference: clean single-speaker speech, 15–30 s, no music
- Chatterbox VC or Seed-VC with preprocessing + best-of-N=3 for best clone quality
- Optional OpenVoice post-step to unify timbre across long files
