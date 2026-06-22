# tensorflow-cpu: ~600 MB vs tensorflow:2.16.1-gpu ~4.8 GB
# build_efficientnetB3.py only defines & saves model architecture - no GPU needed
FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir --retries 5 --timeout 60 \
    dvc[s3]==3.40.0 \
    boto3==1.34.40 \
    tensorflow-cpu==2.16.1 \
    numpy==1.26.4 \
    h5py==3.11.0 \
    tf_keras==2.16.0

COPY Multimodal/models/build_efficientnetB3.py .

ENV TF_USE_LEGACY_KERAS=1
ENV PROCESSED_DIR=/data/processed
ENV MODEL_DIR=/data/model
ENV IMAGE_SIZE=224

COPY .dvc/ .dvc/
CMD ["python", "build_efficientnetB3.py"]
