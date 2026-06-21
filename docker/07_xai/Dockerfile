FROM tensorflow/tensorflow:2.16.1-gpu

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    dvc[s3]>=3.0.0 \
    numpy==1.26.4 \
    pandas==2.2.2 \
    scikit-learn==1.5.0 \
    matplotlib==3.9.0 \
    Pillow==10.3.0 \
    h5py==3.11.0 \
    shap==0.45.1 \
    tqdm==4.66.4 \
    tf_keras==2.16.0

COPY Multimodal/utils/xai.py .
COPY Multimodal/s3_utils.py .

# Buộc TF 2.16 dùng Keras 2 (tf_keras) thay vì Keras 3
ENV TF_USE_LEGACY_KERAS=1

ENV PROCESSED_DIR=/data/processed
ENV SPLITS_DIR=/data/splits
ENV OUTPUT_DIR=/data/output
ENV EVAL_DIR=/data/eval
ENV XAI_DIR=/data/xai
ENV NUM_GRADCAM_SAMPLES=20
ENV NUM_SHAP_SAMPLES=100
ENV SHAP_BACKGROUND=50

COPY .dvc/ .dvc/
CMD ["python", "xai.py"]

