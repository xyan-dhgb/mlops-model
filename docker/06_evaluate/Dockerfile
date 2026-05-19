FROM tensorflow/tensorflow:2.16.1-gpu

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# blinker 1.4 được cài bằng distutils trong base image TF — pip không thể uninstall.
# Gỡ thủ công trước để mlflow cài blinker mới không bị xung đột.
RUN pip install --no-cache-dir --ignore-installed blinker

RUN pip install --no-cache-dir \
    boto3==1.34.131 \
    numpy==1.26.4 \
    pandas==2.2.2 \
    scikit-learn==1.5.0 \
    matplotlib==3.9.0 \
    Pillow==10.3.0 \
    h5py==3.11.0 \
    mlflow==2.13.0 \
    tqdm==4.66.4

COPY Multimodal/training/evaluate.py .
COPY Multimodal/s3_utils.py .

ENV PROCESSED_DIR=/data/processed
ENV SPLITS_DIR=/data/splits
ENV OUTPUT_DIR=/data/output
ENV EVAL_DIR=/data/eval

CMD ["python", "evaluate.py"]
