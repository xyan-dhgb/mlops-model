# Dùng official image của TensorFlow để đảm bảo tương thích 100% CUDA và cuDNN
FROM tensorflow/tensorflow:2.16.1-gpu

WORKDIR /app

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/* \
    && git init

# Cài các thư viện còn lại
# blinker 1.4 được cài bằng distutils trong base image TF — pip không thể uninstall.
# Cài đè thủ công trước để mlflow cài blinker mới không bị xung đột.
RUN pip install --no-cache-dir --ignore-installed blinker
RUN pip install --no-cache-dir \
    
    
    boto3==1.34.40 \
    numpy==1.26.4 \
    pandas==2.2.2 \
    scikit-learn==1.5.0 \
    opencv-python-headless==4.10.0.84 \
    Pillow==10.3.0 \
    tqdm==4.66.4 \
    mlflow>=2.10.0 \
    h5py>=3.9.0 \
    PyYAML>=6.0 \
    matplotlib>=3.7.0 \
    seaborn>=0.13.0 \
    scikit-image>=0.21.0 \
    prometheus_client>=0.17.0 \
    tf_keras==2.16.0

COPY Multimodal/training/train.py .
COPY Multimodal/training/augment.py .
COPY Multimodal/ml_metrics.py .

# Buộc TF 2.16 dùng Keras 2 (tf_keras) — đảm bảo model .h5 được save
# ở format Keras 2, tương thích với evaluate/xai steps.
ENV TF_USE_LEGACY_KERAS=1

ENV PROCESSED_DIR=/app/data/processed
ENV SPLITS_DIR=/app/data/splits
ENV MODEL_DIR=/app/data/model
ENV OUTPUT_DIR=/app/data/output
ENV PHASE1_EPOCHS=20
ENV PHASE2_EPOCHS=10
ENV BATCH_SIZE=32
ENV OVERSAMPLE_RATIO=0.25
ENV CLASS_WEIGHT_MAL=1.2
ENV FINE_TUNE_FROM_LAYER=300
ENV PHASE1_LR=0.001
ENV PHASE2_LR=0.0001
ENV IMAGE_SIZE=224


CMD ["python", "train.py"]
