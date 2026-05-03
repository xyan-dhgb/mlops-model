# ============================================================
# STEP 7 — XAI (Explainable AI)
# • Grad-CAM  : heatmap on last conv layer of EfficientNetB3
# • SHAP      : DeepExplainer on tabular branch
#               → Waterfall plot + Bar (global importance)
# ============================================================
FROM tensorflow/tensorflow:2.16.1-gpu

LABEL stage="07_xai" \
      description="Explainability: Grad-CAM (image) + SHAP DeepExplainer (tabular)"

RUN pip install --no-cache-dir \
        numpy==1.26.4 \
        pandas==2.2.2 \
        shap==0.45.1 \
        matplotlib==3.9.0 \
        opencv-python-headless==4.9.0.80 \
        Pillow==10.3.0 \
        joblib==1.4.2

WORKDIR /app
COPY xai.py focal_loss.py ./

ENV PROCESSED_DIR="/data/processed"
ENV SPLIT_DIR="/data/splits"
ENV OUTPUT_DIR="/data/output"
ENV XAI_DIR="/data/xai"
ENV IMAGE_SIZE="224"
ENV N_SHAP_SAMPLES="100"     # number of samples for global SHAP
ENV LAST_CONV_LAYER="top_conv"  # EfficientNetB3 last conv layer name

RUN mkdir -p /data/xai

CMD ["python", "xai.py"]
