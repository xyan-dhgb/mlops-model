FROM tensorflow/tensorflow:2.16.1-gpu

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
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
