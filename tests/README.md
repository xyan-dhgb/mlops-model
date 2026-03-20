# mlops-model

**MLOps Infrastructure & Multimodal AI for Skin Cancer Diagnosis**
KLTN 2026 — Thiết kế và Triển khai kiến trúc MLOps cho hệ thống AI đa phương thức

---

## Repository Structure

```
mlops-model/
├── .github/workflows/          # CI/CD pipelines (GitHub Actions + OIDC)
├── docker/
│   └── mlflow/
│       ├── docker-compose.yml  # Local MLflow stack (Postgres + MinIO + MLflow)
│       ├── k8s-mlflow.yaml     # Kubernetes manifests (EKS deployment)
│       └── .env.example        # Environment variable template
├── environments/               # Conda / pip environment definitions
├── modules/                    # Shared Python modules (logging, utils)
├── scriptdata/                 # Data extraction & validation notebooks
│   ├── Data_Extraction_Validation_Guide.docx
│   ├── data_extraction_validation.ipynb
│   └── data_extraction_validation_ml.ipynb
├── scripts/
│   └── run_train.py            # Entry point to launch training runs
├── src/                        # Core application source
│   ├── data_preprocessing.py
│   ├── model.py
│   └── train.py
├── tests/                      # Unit tests
│   ├── test_data_preprocessing.py
│   ├── test_model.py
│   └── test_train.py
├── Multimodal/                 # ← Main ML module (see below)
└── multimodal_skin.ipynb       # Exploration notebook
```

---

## Multimodal Module Structure

```
Multimodal/
├── config/
│   └── train_config.yaml       # Training hyperparameters
├── data/
│   └── raw/                    # ISIC 2019/2020 images + CSV (gitignored)
├── data_loader/
│   └── dataloader.py           # build_dataloaders() — train/val splits
├── final/                      # Saved checkpoints + preprocessor artifacts
├── models/
│   └── multimodal_model.py     # EfficientNet-B3 + MetadataMLP + FocalLoss
├── preprocessing/
│   ├── image_preprocessing.py  # Hair removal, color constancy, albumentations
│   └── tabular_preprocessing.py# MetadataPreprocessor, feature engineering
├── training/
│   └── train.py                # Full training loop + MLflow logging
├── utils/                      # Metrics helpers, XRAI/SHAP utilities (WIP)
├── main.py
├── multimodal_skin.ipynb
├── multimodal_skin.py
└── requirement.txt
```

---

## Quick Start

### 1. MLflow Infrastructure (local)

```bash
cd docker/mlflow
cp .env.example .env          # Fill in passwords
docker-compose up -d
# UI: http://localhost:5000
# MinIO console: http://localhost:9001
```

### 2. MLflow on EKS

```bash
kubectl apply -f docker/mlflow/k8s-mlflow.yaml
kubectl port-forward svc/mlflow-service 5000:5000 -n mlops
```

### 3. Train Model

```bash
pip install -r Multimodal/requirement.txt
python Multimodal/training/train.py --config Multimodal/config/train_config.yaml
```

### 4. Run Tests

```bash
pytest tests/ -v
```

---

## Model Architecture

| Component | Choice | Reason |
|---|---|---|
| Image Backbone | EfficientNet-B3 (pretrained) | ~97% ISIC accuracy, lightweight, XRAI-compatible |
| Metadata Branch | MLP 3-layer (age, sex, localization) | Simple late fusion, SHAP-explainable |
| Fusion | Concatenation → FC → Dropout(0.3) | Low overfitting risk |
| Loss | Focal Loss + class weights | Handles melanoma imbalance (~11%) |
| XAI | XRAI (image) + SHAP (metadata) | MDPI 2025 recommendation for dermoscopy |

---

## Dataset

**ISIC 2019/2020** — 10,015 dermoscopic images, 7 classes:
`MEL, NV, BCC, AKIEC, BKL, DF, VASC`

Download from [Kaggle ISIC 2019](https://www.kaggle.com/competitions/siim-isic-melanoma-classification).

---

## References

- MDPI Cosmetics 2025 — EfficientNet-B3 + XRAI for skin cancer
- Nature Communications 2025 — XAI improves dermatologist accuracy (+2.8%)
- Springer ESE Feb 2026 — SecMLOps framework
- EU AI Act 2024 — High-risk AI system requirements
