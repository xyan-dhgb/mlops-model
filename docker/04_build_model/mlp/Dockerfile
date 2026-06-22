FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/* \
    && git init

RUN pip install --no-cache-dir --retries 5 --timeout 60 \
    
    
    boto3==1.34.40 \
    numpy==1.26.4 \
    pandas==2.2.2 \
    scikit-learn==1.5.0 \
    tensorflow-cpu==2.16.1 \
    tf_keras==2.16.0

COPY Multimodal/utils/s3_utils.py /utils/s3_utils.py
COPY Multimodal/models/build_mlp.py .

ENV TF_USE_LEGACY_KERAS=1
ENV PROCESSED_DIR=/app/data/processed
ENV MODEL_DIR=/app/data/model


CMD ["python", "build_mlp.py"]
