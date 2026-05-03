# ============================================================
# STEP 1 — Prepare Data
# Downloads ISIC 2024 image HDF5 + CSV metadata
# ============================================================
FROM python:3.10-slim

LABEL stage="01_prepare_data" \
      description="Download ISIC 2024 dataset (HDF5 images + CSV metadata)"

# System deps for gdown + hdf5 inspection
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        unzip \
        libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Python deps (minimal — only what is needed to download & verify)
RUN pip install --no-cache-dir \
        gdown==5.2.0 \
        h5py==3.11.0 \
        pandas==2.2.2

WORKDIR /app

# Download + unzip script
COPY download_dataset.py .

# Output goes to /data (mount a volume here in production)
RUN mkdir -p /data/raw

ENV GDRIVE_FILE_ID="1Mrr8m5CP-nj9A-d-6AeXlLKjaYY3pAFb"
ENV DATA_DIR="/data/raw"

CMD ["python", "download_dataset.py"]
