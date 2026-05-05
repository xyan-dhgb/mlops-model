FROM python:3.10-slim

# Cài OpenCV cần libGL
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    opencv-python-headless==4.10.0.84 \
    h5py==3.11.0 \
    Pillow==10.3.0 \
    tqdm==4.66.4

COPY preprocess_image.py .

ENV RAW_DIR=/data/raw
ENV PROCESSED_DIR=/data/processed
ENV IMAGE_SIZE=224
ENV MAX_IMAGES=""

CMD ["python", "preprocess_image.py"]
