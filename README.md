# Skin Cancer Multimodal ML – CI/CD Pipeline

> **Dataset**: [Skin Cancer – Kaggle](https://www.kaggle.com/datasets/mahdavi1202/skin-cancer)  
> **Model**: Multimodal CNN (image branch) + Dense (EHR/tabular branch) fused with TensorFlow/Keras  
> **Registry**: AWS ECR  
> **CI/CD**: GitHub Actions

---

## Project Structure

```
mlops-model/
  .github/workflows/
    mlops-pipeline.yml         ← Pipeline CI/CD 6 job
  docker/mlflow/
    docker-compose.yml          ← Stack cục bộ: Postgres + MinIO + MLflow
    k8s-mlflow.yaml             ← Manifest Kubernetes (namespace mlops)
    .env.example                ← Mẫu biến môi trường
  Multimodal/
    config/train_config.yaml    ← Siêu tham số huấn luyện
    data/raw/                   ← Ảnh ISIC + CSV (gitignored)
    data_loader/dataloader.py   ← build_dataloaders()
    final/                      ← Checkpoint + artifact preprocessor
    models/multimodal_model.py  ← EfficientNet-B3 + MLP + FocalLoss
    preprocessing/
      image_preprocessing.py   ← Loại bỏ lông, chuẩn hóa màu, augmentation
      tabular_preprocessing.py ← MetadataPreprocessor, tạo fold
    training/train.py           ← Vòng lặp huấn luyện + ghi log MLflow
    utils/
      xrai_explainer.py         ← XRAI + GradCAM
      shap_explainer.py         ← SHAP + MultimodalXAIRunner
    requirement.txt
  src/
    data_preprocessing.py       ← Xác thực hash + điều phối pipeline (CLI)
    model.py                    ← Nạp mô hình + đánh giá adversarial + MLflow signature
    train.py                    ← Điều phối pipeline đầy đủ
  scripts/
    run_train.py                ← Entry point chính (CLI)
  tests/
    conftest.py
    test_data_preprocessing.py
    test_model.py
    test_train.py
  README.md

```

---

## Pipeline Overview

```
Push / PR
    │
    ▼
┌─────────────────────────────────────────────────────┐
│  JOB 1 – Lint & Static Analysis                     │
│  black, isort, flake8                               │
└──────────────────────┬──────────────────────────────┘
                       │ passes
                       ▼
┌─────────────────────────────────────────────────────┐
│  JOB 2 – Unit Tests (Python 3.10 + 3.11)            │
│  pytest  +  coverage ≥ 75 %                         │
│  Tests for: data_preprocessing / model / train      │
└──────────────────────┬──────────────────────────────┘
                       │ passes  (push only, not PR)
                       ▼
┌─────────────────────────────────────────────────────┐
│  JOB 3 – Build & Push to AWS ECR                    │
│  • Multi-stage Docker build (builder + runtime)     │
│  • Tags: sha-<commit>, <branch>, latest (main only) │
│  • Layer cache via GitHub Actions cache             │
│  • Trivy vulnerability scan                         │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│  JOB 4 – Notify                                     │
│  Slack webhook + GitHub Step Summary                │
└─────────────────────────────────────────────────────┘
```

---

## Required Secrets & Variables

Configure these in **Settings → Secrets and variables → Actions**:

### Secrets (`Settings → Secrets`)

| Secret         | Description                                                                                               |
| -------------- | --------------------------------------------------------------------------------------------------------- |
| `AWS_ROLE_ARN` | ARN of the IAM role GitHub Actions assumes via OIDC (e.g. `arn:aws:iam::123456789:role/GitHubActionsECR`) |

### Variables (`Settings → Variables`)

| Variable            | Example value                                    |
| ------------------- | ------------------------------------------------ |
| `AWS_REGION`        | `ap-southeast-1`                                 |
| `ECR_REGISTRY`      | `123456789.dkr.ecr.ap-southeast-1.amazonaws.com` |
| `SLACK_WEBHOOK_URL` | `https://hooks.slack.com/...` (optional)         |

---

## AWS IAM Setup (OIDC – no long-lived keys)

```bash
# 1. Create OIDC provider for GitHub in AWS
aws iam create-open-id-connect-provider \
  --url https://token.actions.githubusercontent.com \
  --client-id-list sts.amazonaws.com \
  --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1

# 2. Create ECR repository
aws ecr create-repository \
  --repository-name skin-cancer-multimodal \
  --region ap-southeast-1

# 3. Attach a trust policy to the IAM role allowing
#    github.com/<org>/<repo> on branch main/develop
```

**Trust policy for the IAM role** (`trust-policy.json`):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::<ACCOUNT_ID>:oidc-provider/token.actions.githubusercontent.com"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringLike": {
          "token.actions.githubusercontent.com:sub": "repo:<ORG>/<REPO>:ref:refs/heads/*"
        }
      }
    }
  ]
}
```

**Permissions policy** – attach `AmazonEC2ContainerRegistryPowerUser` to the role.

---

## Running Tests Locally

```bash
# Install deps
pip install -r requirements.txt pytest pytest-cov

# Run all unit tests
pytest tests/ -v --cov=src --cov-report=term-missing

# Run a specific test file
pytest tests/test_model.py -v
```

---

## Building the Docker Image Locally

```bash
docker build -f docker/Dockerfile -t skin-cancer-ml:local .

# Train (mount your dataset)
docker run --rm \
  -v /path/to/dataset:/data \
  -v /path/to/output:/app/output \
  -e EPOCHS=5 \
  -e BATCH_SIZE=16 \
  skin-cancer-ml:local
```

---

## Dataset Columns (metadata.csv)

| Column         | Description                       |
| -------------- | --------------------------------- |
| `img_id`       | Image filename                    |
| `diagnostic`   | Target label: BCC / MEL / SCC / … |
| `age`          | Patient age                       |
| `fitspatrick`  | Fitzpatrick skin type (1–6)       |
| `diameter_1/2` | Lesion diameter (mm)              |
| `region`       | Body region                       |
| `gender`       | Patient gender                    |
