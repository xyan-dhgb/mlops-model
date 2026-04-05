# ISIC 2024 – Multimodal Skin Lesion Classifier

Binary classification of skin lesions (Benign vs Malignant) using a
**multimodal deep learning model** that fuses dermoscopy images (CNN branch)
with patient metadata (MLP branch).

Primary competition metric: **pAUC @ 80% TPR** (normalised partial AUC).

---

## Project Structure

```
.github/workflows/ml-ci-cd.yml   ← GitHub Actions CI/CD
docker/
  Dockerfile                     ← Base image
  Dockerfile.preprocessing       ← HDF5 extraction + CSV cleaning
  Dockerfile.training            ← Model training
  Dockerfile.serving             ← FastAPI inference server
  mflow/
    docker-compose.yml           ← MLflow + training + serving stack
    k8s-mflow.yaml               ← Kubernetes manifests
MLflow_signature/
  train.py                       ← MLflow-integrated training entry point
Multimodal/
  config/train_config.yaml       ← All hyperparameters & paths
  data/raw/                      ← ISIC images (HDF5) + CSV metadata
  data_loader/dataloader.py      ← build_dataloaders()
  final/                         ← Saved model + preprocessor
  models/multimodal_model.py     ← build_multimodal_model()
  preprocessing/
    image_preprocessing.py       ← CLAHE, augmentation, HDF5 extraction
    tabular_preprocessing.py     ← Impute, scale, encode
  training/train.py              ← train_model(), evaluate_model(), XAI
  utils/
    metrics.py                   ← pAUC, ROC, metrics table
    predict.py                   ← Single-sample inference helper
    serving.py                   ← FastAPI app
  requirement.txt
scripts/run_train.py             ← CLI entry point
src/
  data_preprocessing.py          ← Public API (re-exports)
  model.py                       ← Public API (re-exports)
tests/
  conftest.py
  test_data_preprocessing.py
  test_model.py
  test_train.py
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r Multimodal/requirement.txt
```

### 2. Prepare data
Place ISIC 2024 files under `Multimodal/data/raw/`:
```
Multimodal/data/raw/
  train-metadata.csv
  train-image.hdf5
```

### 3. Train locally
```bash
# Without MLflow
python scripts/run_train.py --no-mlflow

# With MLflow (requires MLflow server at localhost:5000)
python scripts/run_train.py
```

### 4. Run tests
```bash
pytest tests/ -v --cov=Multimodal --cov=src
```

### 5. Run with Docker Compose
```bash
cd docker/mflow
docker-compose up --build
```

### 6. Inference API
```bash
# After training, start the serving container
docker build -f docker/Dockerfile.serving -t isic2024-serving .
docker run -p 8000:8000 \
  -v $(pwd)/Multimodal/final:/app/Multimodal/final \
  isic2024-serving

# Health check
curl http://localhost:8000/health

# Predict
curl -X POST http://localhost:8000/predict \
  -F "image=@path/to/lesion.jpg" \
  -F 'features={"age_approx": 55, "sex": "male", "anatom_site_general": "torso"}'
```

---

## Model Architecture

```
Image (224×224×3)              Tabular metadata (N features)
       │                                    │
  CNN Branch                           MLP Branch
  Conv2D(32) → BN → Pool               Dense(128) → Dropout(0.3)
  Conv2D(64) → BN → Pool               Dense(64)  → Dropout(0.2)
  Conv2D(128) → BN → Pool              Dense(32)
  GlobalAvgPool → Dense(128)
       │                                    │
       └──────────── Concatenate ───────────┘
                          │
                    Dense(128) → Dropout(0.4)
                    Dense(64)  → Dropout(0.3)
                          │
                  sigmoid (binary) / softmax (multi)
                          │
                       Output
```

---

## XAI (Explainable AI)

Two complementary explanations are generated per prediction:

| Component | Scope | Method |
|---|---|---|
| **Grad-CAM** | Image branch | Gradient-weighted class activation map over the last Conv2D layer |
| **SHAP** | Tabular branch | DeepExplainer waterfall + bar summary (global importance) |

---

## Key Metrics

| Metric | Description |
|---|---|
| **pAUC@80%** | Primary – partial AUC at TPR ≥ 80% |
| AUC-ROC | Overall discriminative ability |
| Accuracy | Overall correctness |
| Sensitivity | True positive rate (recall for malignant) |
| Specificity | True negative rate |

---

## CI/CD Pipeline

```
Push → Lint (Black / Flake8) → Unit Tests → Build Docker images
                                               └─ Deploy to k8s (main only)
                                               └─ Manual training trigger
```
