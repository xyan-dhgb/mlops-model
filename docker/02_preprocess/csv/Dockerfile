FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    dvc[s3]==3.40.0 \
    pathspec==0.11.2 \
    boto3==1.34.40 \
    numpy==1.26.4 \
    pandas==2.2.2 \
    scikit-learn==1.5.0

COPY Multimodal/preprocessing/preprocess_csv.py .

ENV RAW_DIR=/data/raw
ENV PROCESSED_DIR=/data/processed

COPY .dvc/ .dvc/
CMD ["python", "preprocess_csv.py"]
