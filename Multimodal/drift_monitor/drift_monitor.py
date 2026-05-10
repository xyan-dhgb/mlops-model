"""
drift_monitor.py — Giám sát Data Drift trên S3 (chạy liên tục trên EKS)

Đọc từ S3:
  preprocessed/baseline_profile.json       ← profile từ bước evaluate
  preprocessed/production_logs/*.parquet   ← log dự đoán production

Ghi lên S3:
  preprocessed/drift_reports/drift_report_<date>.json
  preprocessed/drift_reports/drift_dashboard_<date>.html  (Evidently)

Metrics: PSI + KS-test mỗi feature, prediction rate drift
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
from scipy import stats

from s3_utils import (
    download_bytes, upload_bytes,
    list_s3_keys,
    S3_OUTPUT_BUCKET,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("drift_monitor")

CHECK_INTERVAL_H = float(os.environ.get("CHECK_INTERVAL_HOURS", "24"))
PSI_THRESHOLD    = float(os.environ.get("PSI_THRESHOLD", "0.25"))
KS_P_THRESHOLD   = float(os.environ.get("KS_P_THRESHOLD", "0.05"))

BASELINE_KEY     = "preprocessed/baseline_profile.json"
PROD_LOG_PREFIX  = "preprocessed/production_logs/"
REPORT_PREFIX    = "preprocessed/drift_reports/"


def compute_psi(base: np.ndarray, prod: np.ndarray, n_bins=10) -> float:
    bins = np.percentile(base, np.linspace(0, 100, n_bins + 1))
    bins[0] -= 1e-8; bins[-1] += 1e-8
    bp = np.histogram(base, bins)[0] / len(base)
    pp = np.histogram(prod, bins)[0] / len(prod)
    bp = np.where(bp == 0, 1e-6, bp)
    pp = np.where(pp == 0, 1e-6, pp)
    return float(round(np.sum((pp - bp) * np.log(pp / bp)), 6))


def severity(psi: float) -> str:
    return "OK" if psi < 0.10 else ("WARNING" if psi < PSI_THRESHOLD else "CRITICAL")


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


def check_drift():
    log.info("=" * 60)
    log.info("Bắt đầu kiểm tra Data Drift...")

    # Load baseline
    try:
        baseline = json.loads(
            download_bytes(BASELINE_KEY, bucket=S3_OUTPUT_BUCKET).decode()
        )
    except Exception:
        log.warning(f"Chưa có baseline tại s3://{S3_OUTPUT_BUCKET}/{BASELINE_KEY}")
        return

    prod_df = load_production_logs()
    if prod_df is None or prod_df.empty:
        log.warning(f"Không có production logs tại s3://{S3_OUTPUT_BUCKET}/{PROD_LOG_PREFIX}")
        return

    log.info(f"Production logs: {len(prod_df):,} records")

    feature_cols = [k for k in baseline if not k.startswith("_")]
    drift_results, critical = {}, []

    for col in feature_cols:
        if col not in prod_df.columns:
            continue
        prod_vals = prod_df[col].dropna().values
        b = baseline[col]

        # Reconstruct baseline sample từ profile (Gaussian approx)
        np.random.seed(42)
        base_vals = np.random.normal(b["mean"], b["std"] + 1e-8, len(prod_vals))

        psi = compute_psi(base_vals, prod_vals)
        ks_stat, ks_pval = stats.ks_2samp(base_vals, prod_vals)
        sev = severity(psi)
        if sev == "CRITICAL":
            critical.append(col)

        drift_results[col] = {
            "psi": psi, "ks_stat": round(float(ks_stat), 6),
            "ks_pval": round(float(ks_pval), 6), "severity": sev,
        }

    # Prediction rate
    pred_drift = False
    if "pred_prob" in prod_df.columns:
        prod_rate = float(prod_df["pred_prob"].mean())
        base_rate = baseline.get("_prediction_rate")
        pred_drift = base_rate is not None and abs(prod_rate - base_rate) > 0.05
        drift_results["_prediction_rate"] = {
            "baseline":   round(base_rate, 4) if base_rate else None,
            "production": round(prod_rate, 4),
            "drift_flag": pred_drift,
        }

    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "timestamp":         date_str,
        "n_production_rows": len(prod_df),
        "psi_threshold":     PSI_THRESHOLD,
        "critical_features": critical,
        "prediction_drift":  pred_drift,
        "overall_status":    "CRITICAL" if (critical or pred_drift) else "OK",
        "features":          drift_results,
    }

    report_key = f"{REPORT_PREFIX}drift_report_{date_str}.json"
    upload_bytes(json.dumps(report, indent=2).encode(), report_key,
                 bucket=S3_OUTPUT_BUCKET)
    log.info(f"Overall: {report['overall_status']}")
    if critical:
        log.warning(f"CRITICAL features: {critical}")
    log.info(f"Report → s3://{S3_OUTPUT_BUCKET}/{report_key}")

    # Evidently HTML
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset

        cols_ok = [c for c in feature_cols if c in prod_df.columns]
        if cols_ok:
            ref = pd.DataFrame({
                c: np.random.normal(baseline[c]["mean"],
                                    baseline[c]["std"] + 1e-8, len(prod_df))
                for c in cols_ok
            })
            evr = Report(metrics=[DataDriftPreset()])
            evr.run(reference_data=ref, current_data=prod_df[cols_ok])
            html_buf = io.StringIO()
            evr.save_html(html_buf)
            html_key = f"{REPORT_PREFIX}drift_dashboard_{date_str}.html"
            upload_bytes(html_buf.getvalue().encode(), html_key,
                         bucket=S3_OUTPUT_BUCKET)
            log.info(f"Evidently → s3://{S3_OUTPUT_BUCKET}/{html_key}")
    except Exception as e:
        log.info(f"Evidently bỏ qua: {e}")

    log.info("Kiểm tra hoàn thành.")


def main():
    log.info(f"Drift Monitor khởi động | Chu kỳ={CHECK_INTERVAL_H}h")
    log.info(f"Bucket: s3://{S3_OUTPUT_BUCKET}/")
    check_drift()
    schedule.every(CHECK_INTERVAL_H).hours.do(check_drift)
    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    main()
