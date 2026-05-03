# ============================================================
# STEP 4 — Build Multimodal Model
# • Image branch : EfficientNetB3 (ImageNet pretrained)
# • Tabular branch : MLP 3 layers (128 → 64 → 32)
# • Fusion Head  : Dense 128 → Dense 64 → Sigmoid
# • Loss         : Focal Loss (gamma=2, alpha=0.25)
# Saves: model architecture JSON + initial weights
# ============================================================
FROM tensorflow/tensorflow:2.16.1-gpu

LABEL stage="04_build_model" \
      description="Construct EfficientNetB3 + MLP multimodal model and save architecture"

RUN pip install --no-cache-dir \
        numpy==1.26.4 \
        pandas==2.2.2 \
        joblib==1.4.2

WORKDIR /app
COPY build_model.py focal_loss.py ./

# Tabular feature dimension comes from preprocessing output
ENV TABULAR_DIM="32"          # override at runtime after inspecting preprocessed data
ENV IMAGE_SIZE="224"
ENV MODEL_DIR="/data/model"
ENV PROCESSED_DIR="/data/processed"

RUN mkdir -p /data/model

CMD ["python", "build_model.py"]
