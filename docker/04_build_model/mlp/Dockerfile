FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    boto3==1.34.131 \
    numpy==1.26.4 \
    pandas==2.2.2 \
    tensorflow-cpu==2.16.1

COPY Multimodal/models/build_mlp.py .
COPY Multimodal/s3_utils.py .

ENV PROCESSED_DIR=/data/processed
ENV MODEL_DIR=/data/model

CMD ["python", "build_mlp.py"]
