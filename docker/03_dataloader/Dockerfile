FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/* \
    && git init

RUN pip install --no-cache-dir --retries 5 --timeout 60 \

    boto3==1.34.40 \
    numpy==1.26.4 \
    pandas==2.2.2 \
    Pillow==10.3.0 \
    scikit-learn==1.5.0 \
    tqdm==4.66.4 \
    opencv-python-headless>=4.8

COPY Multimodal/utils/s3_utils.py /utils/s3_utils.py
COPY Multimodal/data_loader/dataloader.py .

ENV PROCESSED_DIR=/app/data/processed
ENV SPLITS_DIR=/app/data/splits
ENV IMAGE_SIZE=224
ENV RANDOM_SEED=42


CMD ["python", "dataloader.py"]
