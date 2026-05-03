# Data Drift — Nghiên Cứu & Chiến Lược Cho Pipeline ISIC 2024 Multimodal

## 1. Data Drift Là Gì?

**Data drift** (còn gọi là *covariate shift*) xảy ra khi phân phối dữ liệu đầu vào thay đổi theo thời gian, trong khi mối quan hệ giữa đầu vào và nhãn (`p(y|x)`) không đổi. Trong bối cảnh y tế như ISIC 2024, điều này đặc biệt nguy hiểm vì mô hình có thể mất độ chính xác mà không có bất kỳ cảnh báo nào.

### Phân loại Drift

| Loại | Định nghĩa | Ví dụ trong ISIC 2024 |
|------|-----------|----------------------|
| **Covariate Drift** | `p(X)` thay đổi | Thiết bị camera da liễu mới → màu sắc/độ phân giải khác |
| **Label Drift** | `p(y)` thay đổi | Tỷ lệ Malignant tăng theo mùa hoặc dân số |
| **Concept Drift** | `p(y\|X)` thay đổi | Tiêu chuẩn chẩn đoán ung thư da thay đổi |
| **Feature Drift** | Phân phối feature bảng thay đổi | Thay đổi cách đo `age_approx`, `anatom_site` |

---

## 2. Nguồn Gốc Drift Trong Pipeline Multimodal

### 2.1 Image Branch (EfficientNetB3)

**Thiết bị / acquisition drift**
- Bệnh viện dùng dermatoscope khác (Heine vs FotoFinder vs Dermify)
- Ánh sáng, zoom, góc chụp khác nhau giữa các cơ sở
- Nén JPEG ở mức khác nhau → mất chi tiết texture

**Population drift**
- Mô hình train chủ yếu trên dữ liệu da sáng (Fitzpatrick I–III)
- Deploy cho bệnh nhân da sẫm (IV–VI) → feature space khác
- Phân phối độ tuổi, giới tính thay đổi theo cơ sở y tế

**Preprocessing drift**
- CLAHE parameters thay đổi → histogram khác
- Resize khác nhau nếu pipeline tiền xử lý được cập nhật

### 2.2 Tabular / MLP Branch

| Feature | Nguy cơ Drift |
|---------|--------------|
| `age_approx` | Cách ghi nhận tuổi thay đổi (exact vs rounded) |
| `anatom_site_general` | Taxonomy y tế được cập nhật → label khác |
| `sex` | Encoding khác giữa hệ thống HIS |
| Metadata mới | Thêm/xóa cột → dimension mismatch |

### 2.3 Label Drift

- Tỷ lệ Malignant trong dataset ISIC 2024: ~3%
- Trong thực tế lâm sàng: 5–15% tùy cơ sở
- Model đã được train với oversampling (25%) — nếu deploy ratio khác → threshold sai

---

## 3. Phương Pháp Phát Hiện Drift

### 3.1 Statistical Tests Cho Tabular Data

#### Population Stability Index (PSI)
Đo lường sự thay đổi phân phối của một feature:

```
PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)
```

Ngưỡng diễn giải:
- PSI < 0.10 → Không đáng kể
- 0.10 ≤ PSI < 0.25 → Drift nhỏ (theo dõi)
- PSI ≥ 0.25 → Drift nghiêm trọng → cần hành động

#### Kolmogorov-Smirnov Test (K-S Test)
Dùng cho biến liên tục (`age_approx`, scores):
```python
from scipy.stats import ks_2samp
stat, p_value = ks_2samp(train_feature, production_feature)
# p_value < 0.05 → có drift đáng kể
```

#### Chi-Square Test
Dùng cho biến categorical (`sex`, `anatom_site_general`):
```python
from scipy.stats import chi2_contingency
chi2, p, dof, expected = chi2_contingency(contingency_table)
```

#### Wasserstein Distance (Earth Mover's Distance)
Phù hợp nhất cho phân phối liên tục phức tạp:
```python
from scipy.stats import wasserstein_distance
dist = wasserstein_distance(train_dist, prod_dist)
```

### 3.2 Image Drift Detection

#### Embedding-based Drift
Dùng intermediate layer của EfficientNetB3 để lấy embedding:
```python
# Lấy feature từ GlobalAveragePooling layer
embedding_model = tf.keras.Model(
    inputs=model.input,
    outputs=model.get_layer("global_average_pooling2d").output
)
train_embeddings = embedding_model.predict(train_images)
prod_embeddings  = embedding_model.predict(prod_images)
```

Sau đó so sánh bằng:
- Maximum Mean Discrepancy (MMD)
- Fréchet Inception Distance (FID) — phổ biến trong image domain

#### Pixel-level Statistics
Đơn giản nhất, nhanh nhất:
```python
# So sánh mean/std của pixel values
train_mean = train_images.mean(axis=(0,1,2))  # per channel
prod_mean  = prod_images.mean(axis=(0,1,2))
drift_magnitude = np.abs(prod_mean - train_mean).mean()
```

#### Histogram-based Method
```python
import cv2
def bhattacharyya_distance(img1, img2):
    hist1 = cv2.calcHist([img1], [0,1,2], None, [32,32,32], [0,256]*3)
    hist2 = cv2.calcHist([img2], [0,1,2], None, [32,32,32], [0,256]*3)
    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
```

### 3.3 Prediction/Output Drift

Ngay cả khi không có label ground-truth, ta có thể theo dõi:
- Phân phối confidence scores (`prob_malignant`)
- Tỷ lệ dự đoán Malignant theo thời gian
- Entropy của predictions

```python
import numpy as np

def monitor_prediction_drift(new_probs, baseline_probs, threshold=0.05):
    from scipy.stats import ks_2samp
    stat, p = ks_2samp(baseline_probs, new_probs)
    if p < threshold:
        alert = f"DRIFT DETECTED: KS={stat:.3f}, p={p:.4f}"
    else:
        alert = f"OK: KS={stat:.3f}, p={p:.4f}"
    return alert, stat, p
```

### 3.4 Thư Viện Drift Detection Chuyên Dụng

| Thư viện | Ưu điểm | Use case |
|----------|---------|----------|
| **Evidently AI** | Dashboard đẹp, tích hợp ML | Tabular + prediction drift |
| **Alibi Detect** | Nhiều test, hỗ trợ image | Production monitoring |
| **NannyML** | Performance estimation không cần label | Post-deployment |
| **Deepchecks** | End-to-end validation | CI/CD integration |
| **WhyLogs** | Logging profiles nhẹ | Real-time stream |

---

## 4. Chiến Lược Monitoring Drift Trong Pipeline ISIC 2024

### 4.1 Kiến Trúc Monitoring

```
Production Data Stream
        │
        ▼
┌──────────────────┐
│  Feature Logger  │  ← ghi lại X_tab, embedding, pred_prob mỗi inference
└──────────┬───────┘
           │  (batch hàng ngày)
           ▼
┌──────────────────┐
│  Drift Detector  │  ← so sánh với baseline (train distribution)
└──────────┬───────┘
           │
    ┌──────┴──────┐
    │  Alert?     │
    │  PSI>0.25   │
    │  KS p<0.05  │
    └──────┬──────┘
           │ YES
           ▼
┌──────────────────┐
│  Retrain / Tune  │  ← trigger retraining pipeline
└──────────────────┘
```

### 4.2 Baseline Profiles (lưu sau khi train)

```python
import json, numpy as np

def save_baseline_profile(X_tab, feat_names, probs, save_path):
    profile = {
        "feature_stats": {
            feat: {
                "mean": float(X_tab[:, i].mean()),
                "std":  float(X_tab[:, i].std()),
                "p5":   float(np.percentile(X_tab[:, i], 5)),
                "p95":  float(np.percentile(X_tab[:, i], 95)),
            }
            for i, feat in enumerate(feat_names)
        },
        "prediction_stats": {
            "mean_prob": float(probs.mean()),
            "std_prob":  float(probs.std()),
            "malignant_rate": float((probs > 0.5).mean()),
        }
    }
    with open(save_path, "w") as f:
        json.dump(profile, f, indent=2)
```

### 4.3 Evidently AI Dashboard (Khuyến Nghị)

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
report.run(
    reference_data=train_df[feature_cols],
    current_data=production_df[feature_cols],
)
report.save_html("drift_report.html")
```

---

## 5. Ngưỡng Cảnh Báo và Hành Động

| Mức độ | Điều kiện | Hành động |
|--------|----------|-----------|
| 🟢 Bình thường | PSI < 0.10 & KS p > 0.10 | Tiếp tục monitoring |
| 🟡 Cảnh báo | 0.10 ≤ PSI < 0.25 hoặc 0.01 < KS p ≤ 0.10 | Tăng tần suất monitoring, alert team |
| 🔴 Drift nghiêm trọng | PSI ≥ 0.25 hoặc KS p ≤ 0.01 | Dừng deployment, retrain hoặc tune threshold |
| ⚫ Khẩn cấp | Recall Malignant < 0.70 | Tạm dừng model, escalate lên bác sĩ |

---

## 6. Chiến Lược Ứng Phó Drift

### 6.1 Threshold Re-tuning (Nhanh nhất)
Khi chỉ có label drift nhẹ (tỷ lệ Malignant thay đổi):
```python
# Chạy lại grid search threshold trên batch production gần nhất
new_threshold = grid_search_threshold(model, X_img_recent, X_tab_recent, y_recent)
```

### 6.2 Continual Learning / Incremental Retraining
Thêm data production vào tập train định kỳ (hàng tuần/tháng):
```
New Data (production) ──┐
                        ├──► Fine-tune Phase 2 Only (LR=1e-5, 3-5 epochs)
Train Data (old) ───────┘
```
Chỉ fine-tune từ layer 300 trở lên — giữ được knowledge cũ.

### 6.3 Domain Adaptation
Khi deploy sang cơ sở mới với thiết bị hoàn toàn khác:
- **Histogram Matching**: chuẩn hóa màu sắc image sang distribution quen thuộc
- **Style Transfer nhẹ**: dùng Gram matrix loss để align texture
- **CORAL (Correlation Alignment)**: align covariance matrix của tabular features

```python
# Histogram matching đơn giản cho image drift
from skimage.exposure import match_histograms
aligned_img = match_histograms(new_img, reference_img, channel_axis=-1)
```

### 6.4 Ensemble với fallback
Trong production, giữ 2 model:
- `model_current`: model đang dùng
- `model_backup`: model cũ hơn nhưng stable hơn trên domain cũ

Khi drift được phát hiện → tự động switch sang `model_backup` trong khi retrain.

---

## 7. Dockerfile Cho Data Drift Monitoring

```dockerfile
# Dockerfile.drift_monitor
FROM python:3.10-slim

RUN pip install --no-cache-dir \
    evidently==0.4.30 \
    alibi-detect==0.12.0 \
    scipy==1.13.0 \
    pandas==2.2.2 \
    numpy==1.26.4 \
    scikit-learn==1.5.0 \
    matplotlib==3.9.0 \
    schedule==1.2.1

WORKDIR /app
COPY drift_monitor.py baseline_profile.json ./

ENV BASELINE_DIR="/data/baseline"
ENV PRODUCTION_LOG_DIR="/data/production_logs"
ENV ALERT_THRESHOLD_PSI="0.25"
ENV CHECK_INTERVAL_HOURS="24"

CMD ["python", "drift_monitor.py"]
```

---

## 8. Tích Hợp Vào CI/CD

```yaml
# .github/workflows/drift_check.yml
name: Weekly Drift Check
on:
  schedule:
    - cron: '0 8 * * 1'  # Mỗi thứ Hai 8h sáng

jobs:
  drift_check:
    runs-on: ubuntu-latest
    steps:
      - name: Run drift detection
        run: docker compose run drift_monitor
      
      - name: Upload drift report
        uses: actions/upload-artifact@v3
        with:
          name: drift-report
          path: /data/eval/drift_report.html
      
      - name: Alert if critical drift
        if: failure()
        uses: slackapi/slack-github-action@v1
        with:
          payload: '{"text": "🔴 CRITICAL DRIFT DETECTED in ISIC 2024 model!"}'
```

---

## 9. Checklist Triển Khai Production

- [ ] Lưu baseline distribution profiles sau khi train xong (Step 6)
- [ ] Deploy drift monitor container song song với model serving
- [ ] Cấu hình alert (Slack/email) khi PSI > 0.25
- [ ] Lập kế hoạch retrain định kỳ (khuyến nghị: 3 tháng/lần)
- [ ] Version control model + dataset snapshot
- [ ] Log toàn bộ inference (input features + prediction + timestamp)
- [ ] Thiết lập ground-truth feedback loop với bác sĩ (delayed label collection)
- [ ] Test threshold re-tuning trên staging trước khi push production
