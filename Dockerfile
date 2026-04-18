# Voice Changer — Gradio web UI
#
# Slim Python 3.12 base + system libs for audio. PyTorch is installed with
# bundled CUDA 12.1 libraries (only the NVIDIA *driver* has to be passed in
# from the host). For CPU-only builds, pass:
#
#     docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu .
#
# or use the ``app-cpu`` service in docker-compose.yml.
FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HOME=/root/.cache/huggingface \
    TORCH_HOME=/root/.cache/torch \
    XDG_CACHE_HOME=/root/.cache \
    NUMBA_CACHE_DIR=/root/.cache/numba \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7862

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      ffmpeg \
      git \
      libsndfile1 \
      espeak-ng \
      build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch wheels — CUDA 12.1 by default; override with --build-arg for CPU.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
RUN pip install --index-url "${TORCH_INDEX_URL}" \
      "torch>=2.2,<2.6" "torchaudio>=2.2,<2.6"

# Install project requirements. A few entries need special handling:
#   * ``torch`` / ``torchaudio`` — already installed from the CUDA index above
#     (don't let pip replace them with the CPU wheel from PyPI).
#   * ``openvoice-cli`` — every published version pins ``torch==1.13.1`` which
#     conflicts with modern CUDA torch; we install it with ``--no-deps`` and
#     add the real runtime deps by hand.
#   * ``seed-vc`` — best-effort; fall back to the git source, skip if missing.
COPY requirements.txt .
RUN grep -v -E '^\s*(torch|torchaudio|openvoice-cli|seed-vc)(\s*|[<>=!].*)$' \
      requirements.txt > requirements.docker.txt \
 && pip install -r requirements.docker.txt

# openvoice-cli + its real runtime deps (without the bogus torch==1.13.1 pin).
RUN pip install --no-deps openvoice-cli \
 && pip install \
      "cn2an==0.5.22" \
      "eng-to-ipa==0.0.2" \
      "inflect==7.0.0" \
      "jieba==0.42.1" \
      "langid==1.1.6" \
      "pypinyin==0.50.0" \
      "python-dotenv" \
  || echo "[docker] openvoice-cli optional — skipping"

# Seed-VC: try PyPI, fall back to upstream git, otherwise leave the feature
# disabled (app.py detects availability at runtime).
RUN pip install seed-vc \
 || pip install "git+https://github.com/Plachtaa/seed-vc" \
 || echo "[docker] seed-vc optional — skipping"

# Pre-create cache dirs so mounts land on a writable path on first boot.
RUN mkdir -p \
      /root/.cache/huggingface \
      /root/.cache/torch/hub \
      /root/.cache/speechbrain_ecapa \
      /root/.cache/numba \
      /root/.local/share/argos-translate \
      /data

COPY . .

EXPOSE 7862

# Override in compose if you want --low-vram / --cpu / --share etc.
CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "7862", "--low-vram"]
