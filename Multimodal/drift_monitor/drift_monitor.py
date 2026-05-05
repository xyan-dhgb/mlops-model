"""
drift_monitor.py — Container giám sát Data Drift (chạy liên tục)
Chạy định kỳ mỗi CHECK_INTERVAL_HOURS giờ, so sánh production data
với baseline profile được lưu sau bước Evaluate.

Metrics theo dõi:
  - PSI  (Population Stability Index) mỗi feature
  - KS-test (Kolmogorov-Smirnov) cho biến liên tục
  - Prediction rate drift
Đầu vào : /data/eval/baseline_profile.json
           /data/production_logs/*.parquet (hoặc *.csv)
Đầu ra  : /data/xai/drift_reports/drift_report_<date>.json
           /data/xai/drift_reports/drift_report_<date>.html  (Evidently)
"""
import os
import json
import glob
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import schedule
import time
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("drift_monitor")

EVAL_DIR           = os.environ.get("EVAL_DIR", "/data/eval")
PRODUCTION_LOG_DIR = os.environ.get("PRODUCTION_LOG_DIR", "/data/production_logs")
DRIFT_REPORT_DIR   = os.environ.get("DRIFT_REPORT_DIR", "/data/xai/drift_reports")
CHECK_INTERVAL_H   = float(os.environ.get("CHECK_INTERVAL_HOURS", "24"))
PSI_THRESHOLD      = float(os.environ.get("PSI_THRESHOLD", "0.25"))
KS_P_THRESHOLD     = float(os.environ.get("KS_P_THRESHOLD", "0.05"))

BASELINE_PATH = os.path.join(EVAL_DIR, "baseline_profile.json")
os.makedirs(DRIFT_REPORT_DIR, exist_ok=True)


# ── PSI ──────────────────────────────────────────────────────────────────
def compute_psi(baseline_vals: np.ndarray,
                production_vals: np.ndarray,
                n_bins: int = 10) -> float:
    """
    PSI = Σ (Prod% - Base%) × ln(Prod% / Base%)
    PSI < 0.10 → OK | 0.10–0.25 → Theo dõi | ≥ 0.25 → Drift nghiêm trọng
    """
    bins = np.percentile(baseline_vals, np.linspace(0, 100, n_bins + 1))
    bins[0]  -= 1e-8
    bins[-1] += 1e-8

    base_counts = np.histogram(baseline_vals, bins=bins)[0]
    prod_counts = np.histogram(production_vals, bins=bins)[0]

    base_pct = base_counts / base_counts.sum()
    prod_pct = prod_counts / prod_counts.sum()

    # Tránh log(0)
    base_pct = np.where(base_pct == 0, 1e-6, base_pct)
    prod_pct = np.where(prod_pct == 0, 1e-6, prod_pct)

    psi = float(np.sum((prod_pct - base_pct) * np.log(prod_pct / base_pct)))
    return round(psi, 6)


def severity(psi: float) -> str:
    if psi < 0.10:
        return "OK"
    elif psi < 0.25:
        return "WARNING"
    else:
        return "CRITICAL"


# ── Load production logs ─────────────────────────────────────────────────
def load_production_logs() -> pd.DataFrame | None:
    parquet_files = glob.glob(os.path.join(PRODUCTION_LOG_DIR, "*.parquet"))
    csv_files     = glob.glob(os.path.join(PRODUCTION_LOG_DIR, "*.csv"))

    dfs = []
    for fp in parquet_files:
        dfs.append(pd.read_parquet(fp))
    for fp in csv_files:
        dfs.append(pd.read_csv(fp))

    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


# ── Check drift ──────────────────────────────────────────────────────────
def check_drift():
    log.info("=" * 60)
    log.info("Bắt đầu kiểm tra Data Drift...")

    if not os.path.exists(BASELINE_PATH):
        log.warning(f"Chưa có baseline profile tại {BASELINE_PATH}. "
                    "Hãy chạy bước Evaluate trước.")
        return

    with open(BASELINE_PATH) as f:
        baseline = json.load(f)

    prod_df = load_production_logs()
    if prod_df is None or prod_df.empty:
        log.warning(f"Không tìm thấy production logs trong {PRODUCTION_LOG_DIR}")
        return

    log.info(f"Production logs: {len(prod_df)} records")

    feature_cols = [k for k in baseline.keys() if not k.startswith("_")]
    drift_results = {}
    critical_features = []

    for col in feature_cols:
        if col not in prod_df.columns:
            log.warning(f"Feature '{col}' không có trong production logs — bỏ qua")
            continue

        prod_vals = prod_df[col].dropna().values
        base_mean = baseline[col]["mean"]
        base_std  = baseline[col]["std"]

        # Tạo mẫu baseline từ Gaussian (vì không lưu raw values)
        np.random.seed(42)
        base_vals = np.random.normal(base_mean, base_std + 1e-8, len(prod_vals))

        psi = compute_psi(base_vals, prod_vals)
        ks_stat, ks_pval = stats.ks_2samp(base_vals, prod_vals)

        sev = severity(psi)
        if sev == "CRITICAL":
            critical_features.append(col)

        drift_results[col] = {
            "psi":     psi,
            "ks_stat": round(float(ks_stat), 6),
            "ks_pval": round(float(ks_pval), 6),
            "severity": sev,
        }

    # Prediction rate drift
    if "pred_prob" in prod_df.columns:
        pred_rate = float(prod_df["pred_prob"].mean())
        base_rate = baseline.get("_prediction_rate", None)
        pred_drift = abs(pred_rate - base_rate) > 0.05 if base_rate else False
        drift_results["_prediction_rate"] = {
            "baseline": round(base_rate, 4) if base_rate else None,
            "production": round(pred_rate, 4),
            "delta": round(abs(pred_rate - (base_rate or 0)), 4),
            "drift_flag": pred_drift,
        }
    else:
        pred_drift = False

    # Tổng hợp report
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "timestamp":          date_str,
        "n_production_rows":  len(prod_df),
        "psi_threshold":      PSI_THRESHOLD,
        "ks_p_threshold":     KS_P_THRESHOLD,
        "critical_features":  critical_features,
        "prediction_drift":   pred_drift,
        "overall_status":     "CRITICAL" if critical_features or pred_drift else "OK",
        "features":           drift_results,
    }

    report_path = os.path.join(DRIFT_REPORT_DIR, f"drift_report_{date_str}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    log.info(f"Overall status: {report['overall_status']}")
    if critical_features:
        log.warning(f"CRITICAL features: {critical_features}")
    if pred_drift:
        log.warning("Prediction rate drift phát hiện! Kiểm tra model ngay.")
    log.info(f"Drift report → {report_path}")

    # ── Evidently HTML Dashboard (nếu có thư viện) ────────────────────
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset

        feature_cols_present = [c for c in feature_cols if c in prod_df.columns]
        if feature_cols_present:
            base_sample = pd.DataFrame({
                col: np.random.normal(baseline[col]["mean"],
                                       baseline[col]["std"] + 1e-8,
                                       len(prod_df))
                for col in feature_cols_present
            })
            evidently_report = Report(metrics=[DataDriftPreset()])
            evidently_report.run(
                reference_data=base_sample,
                current_data=prod_df[feature_cols_present],
            )
            html_path = os.path.join(DRIFT_REPORT_DIR,
                                     f"drift_dashboard_{date_str}.html")
            evidently_report.save_html(html_path)
            log.info(f"Evidently dashboard → {html_path}")
    except Exception as e:
        log.info(f"Evidently dashboard bỏ qua: {e}")

    log.info("Kiểm tra hoàn thành.")


# ── Main loop ─────────────────────────────────────────────────────────────
def main():
    log.info(f"Drift Monitor khởi động | Chu kỳ: {CHECK_INTERVAL_H}h")

    # Chạy ngay lập tức lần đầu
    check_drift()

    # Lên lịch định kỳ
    schedule.every(CHECK_INTERVAL_H).hours.do(check_drift)

    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    main()
