# ============================================================
# STEP 6 — Evaluate + Threshold Tuning
# Metrics: AUC, pAUC (TPR≥0.80), F1, Recall
# Grid-searches optimal decision threshold on validation set
# Outputs: metrics.json, confusion_matrix.png, roc_curve.png
# ============================================================
FROM tensorflow/tensorflow:2.16.1-gpu

LABEL stage="06_evaluate" \
      description="Evaluate model, tune threshold, compute AUC/pAUC/F1/Recall"

RUN pip install --no-cache-dir \
        numpy==1.26.4 \
        pandas==2.2.2 \
        scikit-learn==1.5.0 \
        matplotlib==3.9.0 \
        seaborn==0.13.2 \
        Pillow==10.3.0 \
        joblib==1.4.2

WORKDIR /app
COPY evaluate.py focal_loss.py ./

ENV PROCESSED_DIR="/data/processed"
ENV SPLIT_DIR="/data/splits"
ENV OUTPUT_DIR="/data/output"
ENV EVAL_DIR="/data/eval"
ENV IMAGE_SIZE="224"
ENV MIN_TPR="0.80"         # pAUC lower bound

RUN mkdir -p /data/eval

CMD ["python", "evaluate.py"]
