FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir --retries 5 --timeout 60 \
    dvc[s3] \
    boto3 \
    s3fs \
    numpy==1.26.4 \
    pandas==2.2.2 \
    scikit-learn==1.5.0 \
    tensorflow-cpu==2.16.1 \
    tf_keras==2.16.0

COPY Multimodal/models/build_mlp.py .

ENV TF_USE_LEGACY_KERAS=1
ENV PROCESSED_DIR=/data/processed
ENV MODEL_DIR=/data/model

COPY .dvc/ .dvc/
CMD ["python", "build_mlp.py"]
