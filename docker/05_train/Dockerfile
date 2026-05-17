# Base image có CUDA 12.3 + cuDNN 9 - tương thích TensorFlow 2.16.1 GPU
# Dùng runtime (nhẹ ~2GB) + cài thêm đúng CUDA compute libs mà TF training cần.
# Không dùng devel (~5GB) để tránh SIGBUS khi extract layer lớn.
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

WORKDIR /app

# Cài Python 3.11 + pip + lib hệ thống
# + CUDA compute libs mà TensorFlow cần để train trên GPU:
#   libcublas   — matrix multiply (forward/backward pass)
#   libcufft    — FFT ops
#   libcurand   — random number generation (dropout, init)
#   libcusolver — linear algebra solvers
#   libcusparse — sparse matrix ops
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    libglib2.0-0 \
    libcublas-12-3 \
    libcufft-12-3 \
    libcurand-12-3 \
    libcusolver-12-3 \
    libcusparse-12-3 \
    && ln -sf python3.11 /usr/bin/python \
    && ln -sf python3.11 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# tensorflow[and-cuda] = GPU build (TF >= 2.12 tách CPU/GPU thành package riêng)
# Không dùng "tensorflow" thuần vì pip sẽ cài bản CPU-only dù có CUDA sẵn.
RUN pip install --no-cache-dir "tensorflow[and-cuda]==2.16.1"

# Cài các thư viện còn lại
RUN pip install --no-cache-dir \
    boto3==1.34.131 \
    numpy==1.26.4 \
    pandas==2.2.2 \
    scikit-learn==1.5.0 \
    opencv-python-headless==4.10.0.84 \
    Pillow==10.3.0 \
    tqdm==4.66.4

COPY Multimodal/training/train.py .
COPY Multimodal/training/augment.py .
COPY Multimodal/s3_utils.py .

ENV PROCESSED_DIR=/data/processed
ENV SPLITS_DIR=/data/splits
ENV MODEL_DIR=/data/model
ENV OUTPUT_DIR=/data/output
ENV PHASE1_EPOCHS=20
ENV PHASE2_EPOCHS=10
ENV BATCH_SIZE=32
ENV OVERSAMPLE_RATIO=0.25
ENV CLASS_WEIGHT_MAL=1.2
ENV FINE_TUNE_FROM_LAYER=300
ENV PHASE1_LR=0.001
ENV PHASE2_LR=0.0001
ENV IMAGE_SIZE=224

CMD ["python", "train.py"]
