FROM tensorflow/tensorflow:2.16.1-gpu

WORKDIR /app

RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    h5py==3.11.0

COPY Multimodal/models/build_efficientnetB3.py .
COPY Multimodal/s3_utils.py .

ENV PROCESSED_DIR=/data/processed
ENV MODEL_DIR=/data/model
ENV IMAGE_SIZE=224

CMD ["python", "build_efficientnetB3.py"]
