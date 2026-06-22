FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    dvc[s3]==3.40.0 \
    pathspec==0.11.2 \
    boto3==1.34.40 \
    h5py==3.11.0 \
    pandas==2.2.2

COPY Multimodal/data/download_dataset.py .

# DATA_DIR and AWS/S3 settings are passed by the runtime environment.
ENV DATA_DIR=/data/raw
ENV AWS_ACCESS_KEY_ID=""
ENV AWS_SECRET_ACCESS_KEY=""
ENV AWS_SESSION_TOKEN=""
ENV AWS_REGION=ap-southeast-1
ENV S3_INPUT_BUCKET=kltn-isic-2024-challenge
ENV S3_INPUT_PREFIX=isic-2024-challenge

COPY .dvc/ .dvc/
CMD ["python", "download_dataset.py"]
