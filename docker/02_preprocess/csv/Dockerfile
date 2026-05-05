FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas==2.2.2 \
    scikit-learn==1.5.0

COPY preprocess_csv.py .

ENV RAW_DIR=/data/raw
ENV PROCESSED_DIR=/data/processed

CMD ["python", "preprocess_csv.py"]
