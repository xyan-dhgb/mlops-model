FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir --retries 5 --timeout 60 \
    dvc[s3] \
    boto3 \
    s3fs \
    numpy==1.26.4 \
    pandas==2.2.2 \
    Pillow==10.3.0 \
    scikit-learn==1.5.0 \
    tqdm==4.66.4 \
    opencv-python-headless>=4.8

COPY Multimodal/data_loader/dataloader.py .

ENV PROCESSED_DIR=/data/processed
ENV SPLITS_DIR=/data/splits
ENV IMAGE_SIZE=224
ENV RANDOM_SEED=42

COPY .dvc/ .dvc/
CMD ["python", "dataloader.py"]
