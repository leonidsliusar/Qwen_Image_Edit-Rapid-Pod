FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=0 \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=0

RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    gcc \
    git \
    libpq-dev \
    libgl1 \
    libglib2.0-0 \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
        python3.11 \
        python3.11-venv \
        python3.11-dev \
        python3.11-distutils \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip

COPY requirements.txt /requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /requirements.txt

WORKDIR /

COPY ./app /app

ENV MODEL_PATH="/runpod-volume/huggingface-cache/hub/"

CMD ["python", "-m", "app.handler"]