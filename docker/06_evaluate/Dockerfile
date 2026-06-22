FROM tensorflow/tensorflow:2.16.1-gpu

WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/* \
    && git init

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# blinker 1.4 được cài bằng distutils trong base image TF — pip không thể uninstall.
# Gỡ thủ công trước để mlflow cài blinker mới không bị xung đột.
RUN pip install --no-cache-dir --ignore-installed blinker

RUN pip install --no-cache-dir \
    dvc[s3]==3.40.0 \
    pathspec==0.11.2 \
    boto3==1.34.40 \
    numpy==1.26.4 \
    pandas==2.2.2 \
    scikit-learn==1.5.0 \
    matplotlib==3.9.0 \
    Pillow==10.3.0 \
    h5py==3.11.0 \
    mlflow==2.13.0 \
    tqdm==4.66.4 \
    tf_keras==2.16.0

COPY Multimodal/training/evaluate.py .
COPY Multimodal/ml_metrics.py .

# Buộc TF 2.16 dùng Keras 2 (tf_keras) thay vì Keras 3 — tránh mọi lỗi
# serialization khi load .h5 (quantization_config, renorm, get_tensor args, ...)
ENV TF_USE_LEGACY_KERAS=1

ENV PROCESSED_DIR=/app/data/processed
ENV SPLITS_DIR=/app/data/splits
ENV OUTPUT_DIR=/app/data/output
ENV EVAL_DIR=/app/data/eval

COPY .dvc/ .dvc/
CMD ["python", "evaluate.py"]

