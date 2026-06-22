FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/* \
    && git init

RUN pip install --no-cache-dir \
    
    
    boto3==1.34.40 \
    h5py==3.11.0 \
    pandas==2.2.2

COPY Multimodal/data/download_dataset.py .

# DATA_DIR and AWS/S3 settings are passed by the runtime environment.
ENV DATA_DIR=/app/data/raw
ENV AWS_ACCESS_KEY_ID=""
ENV AWS_SECRET_ACCESS_KEY=""
ENV AWS_SESSION_TOKEN=""
ENV AWS_REGION=ap-southeast-1
ENV S3_INPUT_BUCKET=kltn-isic-2024-challenge
ENV S3_INPUT_PREFIX=isic-2024-challenge


CMD ["python", "download_dataset.py"]
