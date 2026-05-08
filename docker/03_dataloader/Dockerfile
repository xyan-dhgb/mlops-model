FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas==2.2.2 \
    scikit-learn==1.5.0

COPY Multimodal/data_loader/dataloader.py .

ENV PROCESSED_DIR=/data/processed
ENV SPLITS_DIR=/data/splits
ENV RANDOM_SEED=42

CMD ["python", "dataloader.py"]
