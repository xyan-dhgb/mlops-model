# ISIC 2024 — Multimodal Skin Lesion Classifier | Docker Pipeline

**EfficientNetB3 + MLP + Focal Loss + Two-Phase Training**

---

## Cấu trúc thư mục

```
isic2024_docker/
├── docker-compose.yml
├── .env.example
│
├── 01_prepare_data/
│   ├── Dockerfile
│   └── download_dataset.py
│
├── 02_preprocess/
│   ├── image/
│   │   ├── Dockerfile
│   │   └── preprocess_image.py     # CLAHE + Gaussian + Contrast
│   └── csv/
│       ├── Dockerfile
│       └── preprocess_csv.py       # Impute + Outlier + LabelEnc + Scaler
│
├── 03_dataloader/
│   ├── Dockerfile
│   └── dataloader.py               # Stratified split 64/16/20
│
├── 04_build_model/
│   ├── efficientnetB3/
│   │   ├── Dockerfile              # tensorflow:2.16.1-gpu
│   │   └── build_efficientnetB3.py
│   └── mlp/
│       ├── Dockerfile              # python:3.10-slim + tensorflow-cpu
│       └── build_mlp.py
│
├── 05_train/
│   ├── Dockerfile                  # tensorflow:2.16.1-gpu
│   ├── train.py                    # Two-phase training
│   └── augment.py                  # Augmentation + Oversampling
│
├── 06_evaluate/
│   ├── Dockerfile
│   └── evaluate.py                 # AUC/pAUC/F1 + threshold tuning
│
├── 07_xai/
│   ├── Dockerfile
│   └── xai.py                      # Grad-CAM + SHAP DeepExplainer
│
└── drift_monitor/
    ├── Dockerfile
    └── drift_monitor.py            # PSI + KS-test + Evidently
```

---

## Volumes chia sẻ

| Volume | Nguồn → Đích |
|---|---|
| `data_raw` | prepare_data → preprocess_image, preprocess_csv |
| `data_processed` | preprocess → dataloader, build_model, train, evaluate, xai |
| `data_splits` | dataloader → train, evaluate, xai |
| `data_model` | build_model → train |
| `data_output` | train → evaluate, xai |
| `data_eval` | evaluate → xai, drift_monitor |
| `data_xai` | xai, drift_monitor → output |

---

## Chạy toàn bộ pipeline

```bash
# 1. Cấu hình biến môi trường
cp .env.example .env
# Chỉnh GDRIVE_FILE_ID trong .env

# 2. Chạy toàn bộ pipeline
docker compose up --build

# 3. Chạy từng bước riêng lẻ
docker compose up --build prepare_data
docker compose up --build preprocess_image preprocess_csv
docker compose up --build dataloader
docker compose up --build build_efficientnetB3 build_mlp
docker compose up --build train
docker compose up --build evaluate
docker compose up --build xai

# 4. Drift monitor chạy nền liên tục
docker compose up -d drift_monitor
```

---

## Lưu ý đường dẫn

- Tất cả container đọc/ghi qua **named volumes** — không hard-code path tuyệt đối
- Biến môi trường `*_DIR` kiểm soát toàn bộ đường dẫn, dễ override khi test
- `MAX_IMAGES` trong `preprocess_image` có thể set để chạy nhanh khi debug
