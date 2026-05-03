# ============================================================
# STEP 2 — Preprocess (Image + Tabular)
# • Tabular : impute, encode, scale, outlier removal
# • Image   : CLAHE, Gaussian blur, contrast enhance,
#             extract from HDF5 → PNG on disk
# ============================================================
FROM python:3.10-slim

LABEL stage="02_preprocess" \
      description="Preprocess tabular CSV and extract/preprocess images from HDF5"

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        numpy==1.26.4 \
        pandas==2.2.2 \
        scikit-learn==1.5.0 \
        opencv-python-headless==4.9.0.80 \
        Pillow==10.3.0 \
        h5py==3.11.0 \
        joblib==1.4.2

WORKDIR /app
COPY preprocess.py .

# Mount /data/raw (read) and /data/processed (write)
ENV RAW_DIR="/data/raw"
ENV PROCESSED_DIR="/data/processed"
ENV IMAGE_SIZE="224"
ENV MAX_IMAGES=""     # empty = all images

RUN mkdir -p /data/processed/images

CMD ["python", "preprocess.py"]
