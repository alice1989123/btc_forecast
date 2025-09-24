FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Environment setup
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC
WORKDIR /app

# Install Python 3.11 and system dependencies
# Install Python 3.11 and dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common curl git build-essential \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
        python3.11 python3.11-distutils python3.11-venv \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# (Optional) Reconfirm Python version for debug
RUN python3.11 --version && python3.11 -m pip --version

# Install PyTorch with CUDA 12.1 using Python 3.11
RUN python3.11 -m pip install --upgrade pip
RUN python3.11 -m pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Install other Python dependencies
COPY requirements.txt .
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY btc_forecast .
COPY config .
COPY logs .
COPY generate_predictions.py .
COPY train_models.py .
COPY metadata.py .
COPY utils.py .
COPY download_artifacts.py .

RUN python3.11 -m pip install  psycopg2-binary


# Run script
CMD ["python3.11", "generate_predictions.py", "--model_name=LSTMModel"]
