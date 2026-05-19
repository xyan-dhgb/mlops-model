"""
drift_monitor.py — Giám sát Data Drift trên S3 (chạy liên tục trên EKS)

Logic drift hoàn toàn đồng bộ với Phần 6 của notebook
  multimodal_skin_isic2024_efficientnetB3.ipynb

Đọc từ S3:
  preprocessed/baseline_profile.json       ← profile từ save_baseline_profile()
  preprocessed/production_logs/*.parquet   ← log dự đoán production
  splits/train/X_img_train.npy             ← dùng cho image pixel drift (tuỳ chọn)

Ghi lên S3:
  preprocessed/drift_reports/drift_report_<date>.json
  preprocessed/drift_reports/drift_dashboard_<date>.html  (Evidently)

Metrics (khớp với notebook):
  ─ PSI          : bin theo combined percentile, eps=1e-8   (Phần 6.3)
  ─ KS-test      : scipy.stats.ks_2samp                    (Phần 6.4)
  ─ Wasserstein  : scipy.stats.wasserstein_distance         (Phần 6.4)
  ─ Image Drift  : KS-test + Wasserstein theo từng channel  (Phần 6.5)
  ─ Pred Drift   : KS-test + malignant rate delta           (Phần 6.6)
  ─ Dashboard    : tổng hợp alerts theo mức CRITICAL/WARNING(Phần 6.7)
"""

import io
import json
import logging
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import schedule
from scipy.stats import ks_2samp, wasserstein_distance

from s3_utils import (
    download_bytes,
    list_s3_keys,
    load_npy,
    upload_bytes,
    S3_OUTPUT_BUCKET,
)

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("drift_monitor")

# ── Cấu hình từ env ───────────────────────────────────────────────────────────
CHECK_INTERVAL_H  = float(os.environ.get("CHECK_INTERVAL_HOURS", "24"))
PSI_THRESHOLD     = float(os.environ.get("PSI_THRESHOLD", "0.25"))   # CRITICAL
PSI_WARNING       = float(os.environ.get("PSI_WARNING", "0.10"))     # WARNING
KS_ALPHA          = float(os.environ.get("KS_ALPHA", "0.05"))
PRED_RATE_DELTA   = float(os.environ.get("PRED_RATE_DELTA", "0.05"))
IMG_DRIFT_SAMPLE  = int(os.environ.get("IMG_DRIFT_SAMPLE", "10000"))

# ── Đường dẫn S3 ─────────────────────────────────────────────────────────────
BASELINE_KEY       = "preprocessed/baseline_profile.json"
PROD_LOG_PREFIX    = "preprocessed/production_logs/"
REPORT_PREFIX      = "preprocessed/drift_reports/"
IMG_BASELINE_KEY   = "splits/train/X_img_train.npy"   # tuỳ chọn


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  6.3 — PSI (Population Stability Index)                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def compute_psi(baseline_arr: np.ndarray, current_arr: np.ndarray,
                bins: int = 10, eps: float = 1e-8) -> float:
    """
    PSI dùng bin edge tính từ combined distribution (giống notebook Phần 6.3).
    Ngưỡng: PSI < 0.10 🟢 | 0.10–0.25 🟡 | ≥ 0.25 🔴
    """
    baseline_arr = baseline_arr[~np.isnan(baseline_arr)]
    current_arr  = current_arr[~np.isnan(current_arr)]
    if len(baseline_arr) == 0 or len(current_arr) == 0:
        return 0.0

    combined = np.concatenate([baseline_arr, current_arr])
    edges    = np.unique(np.percentile(combined, np.linspace(0, 100, bins + 1)))
    if len(edges) < 2:
        return 0.0

    base_cnt, _ = np.histogram(baseline_arr, bins=edges)
    curr_cnt, _ = np.histogram(current_arr,  bins=edges)

    base_pct = base_cnt / (base_cnt.sum() + eps) + eps
    curr_pct = curr_cnt / (curr_cnt.sum() + eps) + eps
    return float(np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct)))


def psi_severity(psi_val: float) -> str:
    """Trả về chuỗi mức độ theo ngưỡng notebook."""
    if psi_val < PSI_WARNING:
        return "OK"
    elif psi_val < PSI_THRESHOLD:
        return "WARNING"
    return "CRITICAL"


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  6.4 — Statistical Tests (KS-test + Wasserstein)                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def run_feature_tests(baseline_arr: np.ndarray,
                      current_arr: np.ndarray) -> dict:
    """KS-test + Wasserstein cho một feature (giống notebook Phần 6.4)."""
    baseline_arr = baseline_arr[~np.isnan(baseline_arr)]
    current_arr  = current_arr[~np.isnan(current_arr)]

    psi_val           = compute_psi(baseline_arr, current_arr)
    ks_stat, ks_pval  = ks_2samp(baseline_arr, current_arr)
    w_dist            = wasserstein_distance(baseline_arr, current_arr)
    drift_flag        = bool(ks_pval < KS_ALPHA)
    sev               = psi_severity(psi_val)

    return {
        "psi":         round(float(psi_val),  6),
        "ks_stat":     round(float(ks_stat),  6),
        "ks_pval":     round(float(ks_pval),  6),
        "wasserstein": round(float(w_dist),   6),
        "ks_drift":    drift_flag,
        "severity":    sev,
    }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  6.5 — Image Pixel Drift (KS-test + Wasserstein theo channel)          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def detect_image_pixel_drift(X_img_baseline: np.ndarray,
                              X_img_current: np.ndarray,
                              alpha: float = 0.05) -> dict:
    """
    So sánh phân phối pixel theo từng channel R, G, B bằng KS-test.
    Giống notebook Phần 6.5 — dùng sampling để tiết kiệm bộ nhớ.
    """
    channel_names = ["Red", "Green", "Blue"]
    results = {}
    for c, cname in enumerate(channel_names):
        base_px = X_img_baseline[:, :, :, c].flatten()
        curr_px = X_img_current[:, :, :, c].flatten()
        n_samp  = min(IMG_DRIFT_SAMPLE, len(base_px), len(curr_px))
        base_s  = np.random.choice(base_px, n_samp, replace=False)
        curr_s  = np.random.choice(curr_px, n_samp, replace=False)
        ks_stat, ks_p = ks_2samp(base_s, curr_s)
        w_dist        = wasserstein_distance(base_s, curr_s)
        drift_flag    = bool(ks_p < alpha)
        results[cname] = {
            "ks_stat":     round(float(ks_stat), 6),
            "ks_pval":     round(float(ks_p),    6),
            "wasserstein": round(float(w_dist),  6),
            "drift":       drift_flag,
        }
        status = "DRIFT" if drift_flag else "OK"
        log.info(f"  Image [{cname}]: KS={ks_stat:.4f}  p={ks_p:.4f}  "
                 f"W={w_dist:.4f}  → {status}")
    return results


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  6.6 — Prediction Drift (KS + malignant rate delta)                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def monitor_prediction_drift(baseline_probs: np.ndarray,
                              current_probs: np.ndarray,
                              threshold: float = 0.5) -> dict:
    """
    Giám sát phân phối confidence scores mà không cần ground-truth.
    Giống notebook Phần 6.6.
    """
    ks_stat, ks_p = ks_2samp(baseline_probs, current_probs)
    w_dist        = wasserstein_distance(baseline_probs, current_probs)
    base_rate     = float((baseline_probs >= threshold).mean())
    curr_rate     = float((current_probs  >= threshold).mean())
    rate_diff     = abs(curr_rate - base_rate)
    drift_ks      = bool(ks_p < KS_ALPHA)
    drift_rate    = bool(rate_diff > PRED_RATE_DELTA)

    return {
        "ks_stat":         round(float(ks_stat), 6),
        "ks_pval":         round(float(ks_p),    6),
        "wasserstein":     round(float(w_dist),  6),
        "baseline_rate":   round(base_rate,       4),
        "current_rate":    round(curr_rate,        4),
        "rate_diff":       round(rate_diff,        4),
        "drift_ks":        drift_ks,
        "drift_rate":      drift_rate,
        "overall_drift":   drift_ks or drift_rate,
    }


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  6.7 — Alert & Report tổng hợp                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def build_alerts(feature_results: dict, img_results: dict | None,
                 pred_results: dict | None) -> list[dict]:
    """
    Tổng hợp alerts theo logic Phần 6.7 của notebook.
    """
    alerts = []

    # Tabular PSI
    n_crit = sum(1 for r in feature_results.values() if r["severity"] == "CRITICAL")
    n_warn = sum(1 for r in feature_results.values() if r["severity"] == "WARNING")
    if n_crit > 0:
        alerts.append({"type": "TABULAR_PSI_CRITICAL", "severity": "CRITICAL",
                        "detail": f"{n_crit} feature(s) PSI ≥ {PSI_THRESHOLD}"})
    elif n_warn > 0:
        alerts.append({"type": "TABULAR_PSI_WARNING",  "severity": "WARNING",
                        "detail": f"{n_warn} feature(s) PSI ≥ {PSI_WARNING}"})

    # KS-test trên tabular
    n_ks = sum(1 for r in feature_results.values() if r["ks_drift"])
    n_total = len(feature_results)
    if n_ks > n_total * 0.3:
        alerts.append({"type": "KS_MANY_FEATURES", "severity": "CRITICAL",
                        "detail": f"{n_ks}/{n_total} features drifted (KS)"})
    elif n_ks > 0:
        alerts.append({"type": "KS_SOME_FEATURES", "severity": "WARNING",
                        "detail": f"{n_ks}/{n_total} features drifted (KS)"})

    # Image pixel drift
    if img_results:
        n_img_drift = sum(1 for r in img_results.values() if r.get("drift"))
        if n_img_drift >= 2:
            alerts.append({"type": "IMAGE_PIXEL_DRIFT", "severity": "WARNING",
                            "detail": f"{n_img_drift}/3 channels drifted"})

    # Prediction drift
    if pred_results and pred_results.get("overall_drift"):
        detail_parts = []
        if pred_results["drift_ks"]:
            detail_parts.append(f"KS p={pred_results['ks_pval']:.4f}")
        if pred_results["drift_rate"]:
            detail_parts.append(f"rate Δ={pred_results['rate_diff']*100:.2f}%")
        alerts.append({"type": "PREDICTION_DRIFT", "severity": "WARNING",
                        "detail": ", ".join(detail_parts)})

    return alerts


def overall_status(alerts: list[dict]) -> str:
    sevs = [a["severity"] for a in alerts]
    if "CRITICAL" in sevs:
        return "CRITICAL"
    elif "WARNING" in sevs:
        return "WARNING"
    return "OK"


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Load helpers                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def load_baseline() -> dict | None:
    try:
        raw = download_bytes(BASELINE_KEY, bucket=S3_OUTPUT_BUCKET)
        return json.loads(raw.decode())
    except Exception as e:
        log.warning(f"Chưa có baseline tại s3://{S3_OUTPUT_BUCKET}/{BASELINE_KEY}: {e}")
        return None


def load_production_logs() -> pd.DataFrame | None:
    keys = list_s3_keys(PROD_LOG_PREFIX, bucket=S3_OUTPUT_BUCKET)
    dfs  = []
    for key in keys:
        try:
            data = download_bytes(key, bucket=S3_OUTPUT_BUCKET)
            if key.endswith(".parquet"):
                dfs.append(pd.read_parquet(io.BytesIO(data)))
            elif key.endswith(".csv"):
                dfs.append(pd.read_csv(io.BytesIO(data)))
        except Exception as e:
            log.warning(f"Không đọc được {key}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else None


def load_image_baseline() -> np.ndarray | None:
    """Tải baseline image array từ S3 (tuỳ chọn)."""
    try:
        return load_npy(IMG_BASELINE_KEY, bucket=S3_OUTPUT_BUCKET)
    except Exception:
        return None


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Main drift check                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def check_drift():
    log.info("=" * 65)
    log.info("📊 Bắt đầu kiểm tra Data Drift — ISIC 2024 Multimodal")
    log.info("=" * 65)

    # ── 1. Load baseline ──────────────────────────────────────────────────
    baseline = load_baseline()
    if baseline is None:
        return

    feature_stats = baseline.get("feature_stats", {})
    feature_names = baseline.get("feature_names", list(feature_stats.keys()))
    pred_stats    = baseline.get("prediction_stats")

    log.info(f"Baseline: {baseline.get('n_samples',0):,} samples | "
             f"{baseline.get('n_features',0)} features | "
             f"created {baseline.get('created_at','?')}")

    # ── 2. Load production logs ───────────────────────────────────────────
    prod_df = load_production_logs()
    if prod_df is None or prod_df.empty:
        log.warning(f"Không có production logs tại "
                    f"s3://{S3_OUTPUT_BUCKET}/{PROD_LOG_PREFIX}")
        return
    log.info(f"Production logs: {len(prod_df):,} records")

    # ── 3. Tabular feature drift (PSI + KS + Wasserstein) ─────────────────
    log.info("\n── Tabular Feature Drift (PSI + KS + Wasserstein) ──────────")
    feature_results = {}
    critical_features, warning_features = [], []

    for fname in feature_names:
        if fname not in prod_df.columns or fname not in feature_stats:
            continue

        fstat = feature_stats[fname]
        prod_vals = prod_df[fname].dropna().values

        # Tái tạo baseline sample từ profile
        # Ưu tiên dùng hist_edges/hist_counts nếu có (chính xác hơn Gaussian)
        if "hist_edges" in fstat and "hist_counts" in fstat:
            edges  = np.array(fstat["hist_edges"])
            counts = np.array(fstat["hist_counts"], dtype=float)
            probs  = counts / counts.sum()
            mids   = (edges[:-1] + edges[1:]) / 2
            np.random.seed(42)
            base_vals = np.random.choice(mids, size=len(prod_vals), p=probs)
        else:
            np.random.seed(42)
            base_vals = np.random.normal(
                fstat["mean"], max(fstat["std"], 1e-8), len(prod_vals))

        result = run_feature_tests(base_vals, prod_vals)
        feature_results[fname] = result

        if result["severity"] == "CRITICAL":
            critical_features.append(fname)
        elif result["severity"] == "WARNING":
            warning_features.append(fname)

    n_total = len(feature_results)
    n_drift_ks = sum(1 for r in feature_results.values() if r["ks_drift"])
    log.info(f"  PSI CRITICAL : {len(critical_features)}")
    log.info(f"  PSI WARNING  : {len(warning_features)}")
    log.info(f"  KS drift     : {n_drift_ks}/{n_total} features  (alpha={KS_ALPHA})")
    if critical_features:
        log.warning(f"  CRITICAL features: {critical_features}")

    # ── 4. Image Pixel Drift (Phần 6.5) ──────────────────────────────────
    log.info("\n── Image Pixel Drift (KS + Wasserstein / channel) ──────────")
    img_drift_results = None

    if "pred_prob" in prod_df.columns or "image_array" in prod_df.columns:
        X_img_baseline = load_image_baseline()
        # Nếu production log chứa cột pixel (hiếm) hoặc đã có baseline array
        if X_img_baseline is not None and "image_array" in prod_df.columns:
            try:
                X_img_prod = np.stack(prod_df["image_array"].values)
                img_drift_results = detect_image_pixel_drift(
                    X_img_baseline[:500], X_img_prod[:500])
            except Exception as e:
                log.info(f"  Image drift bỏ qua: {e}")
        elif X_img_baseline is None:
            log.info("  Không có X_img_train.npy trên S3 — bỏ qua image drift")
    else:
        log.info("  Không có cột image_array trong log — bỏ qua image drift")

    # ── 5. Prediction Drift (Phần 6.6) ───────────────────────────────────
    log.info("\n── Prediction Drift (KS + malignant rate delta) ────────────")
    pred_drift_results = None

    if "pred_prob" in prod_df.columns and pred_stats is not None:
        prod_probs  = prod_df["pred_prob"].dropna().values
        base_counts = np.array(pred_stats["hist_counts"], dtype=float)
        base_probs_norm = base_counts / base_counts.sum()
        bin_edges   = np.linspace(0, 1, len(base_counts) + 1)
        bin_mids    = (bin_edges[:-1] + bin_edges[1:]) / 2
        np.random.seed(42)
        baseline_probs = np.random.choice(bin_mids, size=len(prod_probs),
                                           p=base_probs_norm)
        threshold = float(os.environ.get("PRED_THRESHOLD", "0.5"))
        pred_drift_results = monitor_prediction_drift(baseline_probs, prod_probs,
                                                       threshold=threshold)
        log.info(f"  Baseline malignant rate: {pred_drift_results['baseline_rate']*100:.2f}%")
        log.info(f"  Production malignant rate: {pred_drift_results['current_rate']*100:.2f}%")
        log.info(f"  Rate delta: {pred_drift_results['rate_diff']*100:.2f}%  "
                 f"KS p={pred_drift_results['ks_pval']:.4f}")
        log.info(f"  Prediction drift: {'YES 🔴' if pred_drift_results['overall_drift'] else 'NO 🟢'}")
    else:
        log.info("  Không có pred_prob hoặc prediction_stats — bỏ qua")

    # ── 6. Tổng hợp alerts & report (Phần 6.7) ───────────────────────────
    alerts = build_alerts(feature_results, img_drift_results, pred_drift_results)
    status = overall_status(alerts)

    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "timestamp":          date_str,
        "baseline_created":   baseline.get("created_at"),
        "n_production_rows":  len(prod_df),
        "n_features_checked": n_total,
        "thresholds": {
            "psi_warning":   PSI_WARNING,
            "psi_critical":  PSI_THRESHOLD,
            "ks_alpha":      KS_ALPHA,
            "pred_rate_delta": PRED_RATE_DELTA,
        },
        "overall_status":     status,
        "alerts":             alerts,
        "summary": {
            "psi_critical_features":  critical_features,
            "psi_warning_features":   warning_features,
            "ks_drifted_count":       n_drift_ks,
            "ks_total_count":         n_total,
        },
        "features":           feature_results,
        "image_drift":        img_drift_results,
        "prediction_drift":   pred_drift_results,
    }

    report_key = f"{REPORT_PREFIX}drift_report_{date_str}.json"
    upload_bytes(json.dumps(report, indent=2).encode(),
                 report_key, bucket=S3_OUTPUT_BUCKET)
    log.info(f"\n{'='*65}")
    log.info(f"  TỔNG KẾT: {status}")
    log.info(f"  Alerts  : {len(alerts)}")
    log.info(f"{'='*65}")
    log.info(f"Report → s3://{S3_OUTPUT_BUCKET}/{report_key}")

    # ── 7. Evidently HTML Dashboard ───────────────────────────────────────
    _try_evidently_dashboard(baseline, feature_names, feature_stats,
                              prod_df, date_str)

    log.info("Kiểm tra hoàn thành.")


def _try_evidently_dashboard(baseline: dict, feature_names: list,
                              feature_stats: dict, prod_df: pd.DataFrame,
                              date_str: str):
    """Tạo Evidently HTML dashboard (Phần 6.7) — bỏ qua nếu không cài."""
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset

        cols_ok = [c for c in feature_names
                   if c in prod_df.columns and c in feature_stats]
        if not cols_ok:
            return

        # Tái tạo reference DataFrame từ baseline profile
        ref_data = {}
        for fname in cols_ok:
            fstat = feature_stats[fname]
            if "hist_edges" in fstat and "hist_counts" in fstat:
                edges  = np.array(fstat["hist_edges"])
                counts = np.array(fstat["hist_counts"], dtype=float)
                probs  = counts / counts.sum()
                mids   = (edges[:-1] + edges[1:]) / 2
                ref_data[fname] = np.random.choice(
                    mids, size=len(prod_df), p=probs)
            else:
                ref_data[fname] = np.random.normal(
                    fstat["mean"], max(fstat["std"], 1e-8), len(prod_df))

        ref_df  = pd.DataFrame(ref_data)
        evr     = Report(metrics=[DataDriftPreset()])
        evr.run(reference_data=ref_df, current_data=prod_df[cols_ok])

        html_buf = io.StringIO()
        evr.save_html(html_buf)
        html_key = f"{REPORT_PREFIX}drift_dashboard_{date_str}.html"
        upload_bytes(html_buf.getvalue().encode(), html_key,
                     bucket=S3_OUTPUT_BUCKET)
        log.info(f"Evidently → s3://{S3_OUTPUT_BUCKET}/{html_key}")
    except Exception as e:
        log.info(f"Evidently bỏ qua: {e}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Entry point                                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    log.info(f"Drift Monitor khởi động | Chu kỳ={CHECK_INTERVAL_H}h")
    log.info(f"Bucket: s3://{S3_OUTPUT_BUCKET}/")
    log.info(f"PSI ngưỡng: WARNING≥{PSI_WARNING} | CRITICAL≥{PSI_THRESHOLD}")
    log.info(f"KS alpha={KS_ALPHA} | Pred rate delta={PRED_RATE_DELTA}")
    check_drift()
    schedule.every(CHECK_INTERVAL_H).hours.do(check_drift)
    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    main()