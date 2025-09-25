FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Env & working dir
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Etc/UTC \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

# System deps + Python 3.11
RUN apt-get update && apt-get install -y \
      software-properties-common curl git build-essential \
      python3.11 python3.11-distutils python3.11-venv \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create & activate virtualenv (no system packages)
RUN python3.11 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# (optional) show versions
RUN python --version && pip --version

# Upgrade pip inside venv
RUN pip install --upgrade pip

# --- Install Python deps in the venv ---
# 1) Torch CUDA 12.1 wheels from PyTorch index
RUN pip install \
      torch==2.5.1+cu121 \
      torchvision==0.20.1+cu121 \
      torchaudio==2.5.1+cu121 \
      --extra-index-url https://download.pytorch.org/whl/cu121

# 2) App requirements
#    (copy requirements first to leverage cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3) If psycopg2-binary isn’t in requirements.txt, install it here
#    (or add it to requirements.txt and delete this)
RUN pip install psycopg2-binary

# Copy source code
COPY btc_forecast ./btc_forecast
COPY config ./config
COPY logs ./logs
COPY generate_predictions.py .
COPY train_models.py .
COPY metadata.py .
COPY utils.py .
COPY download_artifacts.py .

# Default command
CMD ["python", "generate_predictions.py", "--model_name=LSTMModel"]
