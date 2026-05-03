# ============================================================
# STEP 3 — Dataloader
# Stratified split 64% / 16% / 20% for image & tabular,
# serialises splits to disk for downstream steps.
# ============================================================
FROM python:3.10-slim

LABEL stage="03_dataloader" \
      description="Stratified train/val/test split + dataloader construction"

RUN pip install --no-cache-dir \
        numpy==1.26.4 \
        pandas==2.2.2 \
        scikit-learn==1.5.0 \
        joblib==1.4.2

WORKDIR /app
COPY dataloader.py .

ENV PROCESSED_DIR="/data/processed"
ENV SPLIT_DIR="/data/splits"
ENV TRAIN_RATIO="0.64"
ENV VAL_RATIO="0.16"
# TEST_RATIO is implicitly 1 - TRAIN - VAL = 0.20
ENV RANDOM_SEED="42"

RUN mkdir -p /data/splits

CMD ["python", "dataloader.py"]
