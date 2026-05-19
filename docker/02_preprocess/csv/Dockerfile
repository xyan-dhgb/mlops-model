FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    boto3==1.34.131 \
    numpy==1.26.4 \
    pandas==2.2.2 \
    scikit-learn==1.5.0

COPY Multimodal/preprocessing/preprocess_csv.py .
COPY Multimodal/s3_utils.py .

ENV RAW_DIR=/data/raw
ENV PROCESSED_DIR=/data/processed

CMD ["python", "preprocess_csv.py"]
