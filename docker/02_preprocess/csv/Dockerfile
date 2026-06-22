FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/* \
    && git init

RUN pip install --no-cache-dir \
    
    
    boto3==1.34.40 \
    numpy==1.26.4 \
    pandas==2.2.2 \
    scikit-learn==1.5.0

COPY Multimodal/preprocessing/preprocess_csv.py .

ENV RAW_DIR=/app/data/raw
ENV PROCESSED_DIR=/app/data/processed


CMD ["python", "preprocess_csv.py"]
