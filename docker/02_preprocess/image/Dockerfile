FROM python:3.10-slim

# libglib2.0-0 needed by OpenCV headless; retry loop handles transient apt network errors
RUN for i in 1 2 3; do \
        apt-get update && \
        apt-get install -y --no-install-recommends libglib2.0-0 && \
        rm -rf /var/lib/apt/lists/* && break || \
        { echo "apt attempt $i failed, retrying in 10s..."; sleep 10; }; \
    done

WORKDIR /app

RUN pip install --no-cache-dir --retries 5 --timeout 60 \
    dvc[s3]>=3.0.0 \
    boto3>=1.34.0 \
    s3fs>=2024.2.0 \
    numpy==1.26.4 \
    pandas==2.2.2 \
    opencv-python-headless==4.10.0.84 \
    h5py==3.11.0 \
    Pillow==10.3.0 \
    tqdm==4.66.4

COPY Multimodal/preprocessing/preprocess_image.py .

ENV RAW_DIR=/data/raw
ENV PROCESSED_DIR=/data/processed
ENV IMAGE_SIZE=224
ENV MAX_IMAGES=""

COPY .dvc/ .dvc/
CMD ["python", "preprocess_image.py"]
