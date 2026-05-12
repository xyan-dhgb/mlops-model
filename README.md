# ISIC 2024 – Multimodal Skin Lesion Classifier (MLOps)

> EfficientNetB3 + MLP + Focal Loss + Two-Phase Training  
> AUC-ROC: 0.8598 | pAUC: 3.3871 | F1: 0.5587 | Recall: 0.6329 | threshold=0.55

---

## Architecture

```
EfficientNetB3 Branch (Image)          MLP Branch (Tabular / 37 features)
  pretrained ImageNet, top=False         Dense(128, ReLU, Dropout 0.3)
  GlobalAveragePooling2D → [1536]        Dense(64,  ReLU, Dropout 0.2)
  BatchNorm → Dense(256, Dropout 0.4)    Dense(32,  ReLU)
  Dense(128, Dropout 0.3) → [128]        → [32]
           ↘                           ↙
            Concatenate [128 + 32 = 160]
            Fusion Head
              Dense(128, ReLU, Dropout 0.4)
              Dense(64,  ReLU, Dropout 0.3)
              Dense(1, sigmoid) → P(Malignant) ∈ [0, 1]
```

### Two-Phase Training

| | Phase 1 – Head Training | Phase 2 – Fine-Tuning |
|---|---|---|
| Backbone | FROZEN 100% | Unfreeze from layer 300+ |
| LR | 1e-3 (Adam) | 1e-4 |
| Loss | Focal Loss γ=2, α=0.75 | same |
| Monitor | val_auc (maximize) | val_auc (maximize) |
| Patience | 5 | 7 |
| Max Epochs | 15 | 15 |

---

## Project Structure

```
isic2024-mlops/
├── .github/
│   └── workflows/
│       └── ml-ci-cd.yml          # CI/CD: lint → test → build → push → deploy
├── Docker/
│   ├── Dockerfile                # Base image
│   ├── Dockerfile.preprocessing  # Stage 1: HDF5 extract + tabular prep
│   ├── Dockerfile.training       # Stage 2: Two-phase GPU training
│   ├── Dockerfile.serving        # Stage 3: FastAPI inference API
│   └── mlflow/
│       └── docker-compose.yml    # Full local stack
├── MLflow_signature/
│   └── train.py                  # MLflow-tracked training entry-point
├── scripts/
│   ├── preprocess.py             # Preprocessing Docker entrypoint
│   └── run_train.py              # Training Docker entrypoint
├── serving/
│   └── app.py                    # FastAPI app (predict + Grad-CAM)
├── src/
│   ├── data_preprocessing.py     # Image + tabular pipeline
│   └── model.py                  # Model, loss, trainer, evaluator
├── tests/
│   ├── test_data_preprocessing.py
│   ├── test_model.py
│   └── test_train.py
├── requirements.txt
├── requirements-preprocessing.txt
├── requirements-serving.txt
└── requirements-dev.txt
```

---

## Quick Start

### 1. Local development

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

### 2. Full Docker stack (local)

```bash
# Start MLflow tracker
docker compose -f Docker/mlflow/docker-compose.yml up mlflow -d

# Stage 1: Preprocess
docker compose -f Docker/mlflow/docker-compose.yml \
  --profile preprocessing run --rm preprocessing

# Stage 2: Train (requires NVIDIA GPU + nvidia-container-toolkit)
docker compose -f Docker/mlflow/docker-compose.yml \
  --profile training run --rm training

# Stage 3: Serve
docker compose -f Docker/mlflow/docker-compose.yml \
  --profile serving up serving -d
```

### 3. Standalone training

```bash
python MLflow_signature/train.py \
  --csv_path   data/train-metadata.csv \
  --hdf5_path  data/train-image.hdf5  \
  --image_dir  data/images            \
  --output_dir models
```

### 4. Single prediction

```bash
curl -X POST http://localhost:8080/predict \
  -F "image=@lesion.jpg"                   \
  -F 'metadata={"age_approx":55,"sex":"male","anatom_site_general":"trunk"}'
```

---

## Imbalance Handling (~97% Benign / ~3% Malignant)

| Layer | Strategy |
|---|---|
| 1 | Oversampling (Malignant ×strong_aug, target_ratio=0.25) |
| 2 | Class weight ×1.2 for Malignant (avoids triple-penalty bias) |
| 3 | Focal Loss FL = −α(1−p)^γ log(p), γ=2.0, α=0.75 |

---

## Explainability (XAI)

- **Grad-CAM** – heatmap overlay on lesion image (served inline via `/predict`)
- **SHAP DeepExplainer** – waterfall plot for tabular features  
- **`MultimodalXAIRunner`** – unified runner returning both explanations

---

## CI/CD Pipeline

```
push → lint (flake8/mypy)
     → unit tests (pytest)
     → integration smoke-test
     → docker build (preprocessing / training / serving)
     → push to GHCR  [main only]
     → hand off image tag to the deployment repository  [main only]
```

---

## Key Results (Best Phase 2 Epoch)

```
AUC-ROC : 0.8598
pAUC    : 3.3871  (TPR ≥ 80%, ISIC 2024 primary metric)
F1      : 0.5587  (Malignant)
Recall  : 0.6329  (Malignant)
Threshold: 0.55
```
