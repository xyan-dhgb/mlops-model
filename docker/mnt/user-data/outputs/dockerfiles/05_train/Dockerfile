# ============================================================
# STEP 5 — Two-Phase Training
# Phase 1 : Train Fusion Head  (EfficientNetB3 frozen)
#           Focal Loss + Oversampling + class_weight ×1.2
#           LR = 1e-3 | Epochs = 20 | EarlyStopping (patience=5)
# Phase 2 : Fine-tune EfficientNetB3 from layer 300
#           LR = 1e-4 | Epochs = 10
# Saves   : best_model.h5 (checkpoint) + training_history.json
# ============================================================
FROM tensorflow/tensorflow:2.16.1-gpu

LABEL stage="05_train" \
      description="Two-phase training: head training then EfficientNetB3 fine-tuning"

RUN pip install --no-cache-dir \
        numpy==1.26.4 \
        pandas==2.2.2 \
        scikit-learn==1.5.0 \
        Pillow==10.3.0 \
        opencv-python-headless==4.9.0.80 \
        joblib==1.4.2

WORKDIR /app
COPY train.py focal_loss.py augment.py ./

ENV PROCESSED_DIR="/data/processed"
ENV SPLIT_DIR="/data/splits"
ENV MODEL_DIR="/data/model"
ENV OUTPUT_DIR="/data/output"

# Training hyper-parameters (override via -e flags)
ENV PHASE1_EPOCHS="20"
ENV PHASE2_EPOCHS="10"
ENV BATCH_SIZE="32"
ENV OVERSAMPLE_RATIO="0.25"
ENV CLASS_WEIGHT_MAL="1.2"
ENV FINE_TUNE_FROM_LAYER="300"
ENV PHASE1_LR="1e-3"
ENV PHASE2_LR="1e-4"
ENV IMAGE_SIZE="224"

RUN mkdir -p /data/output

CMD ["python", "train.py"]
