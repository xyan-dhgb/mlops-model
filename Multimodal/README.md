# ISIC 2024 — Multimodal Skin Lesion Classification

**EfficientNetB3** (image) + **Tabular MLP** (metadata) → Binary classification  
**Benign (0)** / **Malignant (1)**

---

## Project Structure

```
Multimodal/
├── config/
│   └── train_config.yaml         ← All hyperparameters & paths
├── data/
│   └── raw/                      ← Place ISIC images + CSV here
│       ├── train-metadata.csv
│       ├── train-image.hdf5
│       └── images/               ← Auto-extracted from HDF5
├── data_loader/
│   └── dataloader.py             ← build_dataloaders(cfg)
├── final/                        ← Saved checkpoints + preprocessor
│   ├── best_model_phase1.h5
│   ├── best_model_isic2024.h5
│   ├── scaler.pkl
│   ├── cat_encoders.pkl
│   ├── label_encoder.pkl
│   └── feature_names.pkl
├── models/
│   └── multimodal_model.py       ← build_multimodal_model()
├── preprocessing/
│   ├── image_preprocessing.py    ← CLAHE, augmentation, oversampling
│   └── tabular_preprocessing.py  ← clean, encode, scale
├── training/
│   └── train.py                  ← 2-phase training loop
├── utils/
│   ├── metrics.py                ← pAUC, threshold tuning, evaluation
│   ├── xai.py                    ← Grad-CAM + SHAP
│   └── inference.py              ← predict_single(), predict_from_csv()
├── main.py                       ← Entry point
└── requirement.txt
```

---

## Installation

```bash
pip install -r requirement.txt
```

Requires **Python ≥ 3.10** and **TensorFlow ≥ 2.12**.

---

## Data Setup

1. Download the [ISIC 2024 Challenge dataset](https://challenge.isic-archive.com/landing/2024/)
2. Place files in `data/raw/`:
   ```
   data/raw/train-metadata.csv
   data/raw/train-image.hdf5
   ```
3. Images are automatically extracted from the HDF5 to `data/raw/images/` on first run.

---

## Training

```bash
# Full 2-phase training
python main.py

# Custom config
python main.py --config config/train_config.yaml
```

### Two-Phase Strategy

| Phase | Backbone | LR | Monitor |
|-------|----------|----|---------|
| 1 — Head only | ❄️ Frozen | 1e-3 | val_auc |
| 2 — Fine-tune | 🔥 Layers 300+ | 1e-4 | val_auc |

> **Why not Accuracy?**  
> The dataset is ~97% Benign / ~3% Malignant. A model predicting "all Benign"
> achieves 97% accuracy but 0% Recall on Malignant. We use **AUC**, **pAUC**,
> **Recall**, and **F1** instead.

---

## Model Architecture

```
Image (224×224×3)
    └─ EfficientNetB3 (ImageNet)
       └─ GlobalAvgPool → BN → Dense(256) → Dense(128)
                                                        ─┐
Tabular (N features)                                     ├─ Concatenate
    └─ Dense(128) → BN → Dense(64) → Dense(32)         ─┘
                                                         │
                                                    Dense(256) → Dense(128)
                                                    → Dense(64) → Dense(1, sigmoid)
```

**Loss**: Focal Loss (`γ=2.0, α=0.75`)  
**Imbalance**: Oversampling (target_ratio=0.25) + class_weight ×1.2

---

## Evaluation

```bash
# Eval on saved checkpoint
python main.py --eval-only --checkpoint final/best_model_isic2024.h5
```

Primary metric: **pAUC @ TPR ≥ 0.80** (ISIC 2024 competition metric)

---

## Inference (Single Patient)

```python
from utils.inference import load_model_for_inference, predict_single, visualize_prediction

# Load artifacts
artifacts = load_model_for_inference(
    model_path="final/best_model_isic2024.h5",
    artifacts_dir="final",
)

# Predict
import pandas as pd
df = pd.read_csv("data/raw/train-metadata.csv")
result = predict_single(
    image_path="data/raw/images/ISIC_0015657.jpg",
    metadata_row=df.iloc[0],
    artifacts=artifacts,
    threshold=0.35,   # use threshold from training
)
print(result)
# {'diagnosis': 'Benign (Lành tính)', 'confidence': 94.2, ...}

# Visualise
visualize_prediction("data/raw/images/ISIC_0015657.jpg", result)
```

---

## Explainability (XAI)

```python
from utils.xai import MultimodalXAIRunner

runner = MultimodalXAIRunner(
    model=artifacts["model"],
    background_tabular=X_tab_train[:100],
    background_image=X_img_train[0],
    feature_names=artifacts["feature_names"],
    last_conv_layer="top_activation",   # EfficientNetB3 last conv
)

# Explain one sample (renders Grad-CAM + SHAP waterfall)
result = runner.explain(X_tab_train[0], X_img_train[0], visualize=True)

# Global feature importance
runner.global_importance(X_tab_train, top_k=15, n_sample=100)
```

---

## Configuration Reference (`config/train_config.yaml`)

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `data` | `image_size` | `[224, 224]` | Input resolution |
| `imbalance` | `oversample_ratio` | `0.25` | Target Malignant fraction |
| `imbalance` | `class_weight_multiplier` | `1.2` | Extra weight on Malignant |
| `loss` | `focal_gamma` | `2.0` | Focal Loss γ |
| `loss` | `focal_alpha` | `0.75` | Focal Loss α (Malignant weight) |
| `training.phase1` | `epochs` | `30` | Max epochs Phase 1 |
| `training.phase2` | `fine_tune_from` | `300` | Unfreeze from layer index |
| `evaluation` | `threshold_strategy` | `f1` | Threshold search metric |
| `evaluation` | `min_recall` | `0.60` | Minimum recall constraint |

---

## Key Design Decisions

- **EfficientNetB3 over B0**: +4.5pp ImageNet accuracy, better feature extraction for dermoscopy
- **Focal Loss α=0.75**: Higher than default (0.25) to compensate for extreme class imbalance
- **Oversampling ×1.2 class weight** (not ×1.5): triple-penalty (Focal + oversample + high weight) causes the model to over-predict Malignant and collapse Precision
- **Threshold tuning**: Medical context requires high sensitivity; optimal threshold is typically 0.30–0.45, not the default 0.50
- **pAUC @ TPR≥0.80**: The ISIC competition metric ensures the model is useful in clinical settings where missing a malignancy is costly
