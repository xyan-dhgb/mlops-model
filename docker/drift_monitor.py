"""
drift_monitor.py — Data Drift Monitor cho ISIC 2024 Multimodal Pipeline.

Chạy định kỳ (mặc định mỗi 24h), so sánh production data với baseline.
Outputs: drift_report_<date>.json + drift_report_<date>.html (Evidently)

Env vars:
  BASELINE_DIR          : thư mục chứa baseline_profile.json (từ Step 6)
  PRODUCTION_LOG_DIR    : thư mục chứa production inference logs (parquet/csv)
  ALERT_THRESHOLD_PSI   : PSI ngưỡng cảnh báo (mặc định 0.25)
  CHECK_INTERVAL_HOURS  : tần suất kiểm tra (mặc định 24)
"""

import os, json, logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency, wasserstein_distance

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO
)
log = logging.getLogger(__name__)

BASELINE_DIR    = os.environ.get("BASELINE_DIR",         "/data/baseline")
PROD_LOG_DIR    = os.environ.get("PRODUCTION_LOG_DIR",   "/data/production_logs")
PSI_THRESHOLD   = float(os.environ.get("ALERT_THRESHOLD_PSI", 0.25))
CHECK_INTERVAL  = int(os.environ.get("CHECK_INTERVAL_HOURS",  24))


# ── PSI ──────────────────────────────────────────────────
def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index."""
    eps = 1e-8
    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max())
    breakpoints = np.linspace(min_val, max_val, bins + 1)

    expected_pct = np.histogram(expected, breakpoints)[0] / len(expected) + eps
    actual_pct   = np.histogram(actual,   breakpoints)[0] / len(actual)   + eps

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


# ── Load baseline ─────────────────────────────────────────
def load_baseline() -> dict:
    path = Path(BASELINE_DIR) / "baseline_profile.json"
    if not path.exists():
        raise FileNotFoundError(f"Baseline profile not found: {path}")
    with open(path) as f:
        return json.load(f)


# ── Load recent production logs ───────────────────────────
def load_production_data(lookback_days: int = 7) -> pd.DataFrame:
    """Loads the last `lookback_days` days of inference logs."""
    log_files = sorted(Path(PROD_LOG_DIR).glob("*.parquet"))
    if not log_files:
        log_files = sorted(Path(PROD_LOG_DIR).glob("*.csv"))

    if not log_files:
        raise FileNotFoundError(f"No production logs in {PROD_LOG_DIR}")

    dfs = []
    for f in log_files[-lookback_days:]:
        dfs.append(pd.read_parquet(f) if str(f).endswith(".parquet")
                   else pd.read_csv(f))
    return pd.concat(dfs, ignore_index=True)


# ── Drift report ──────────────────────────────────────────
def run_drift_check() -> dict:
    baseline = load_baseline()
    prod_df  = load_production_data()

    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "n_production_samples": len(prod_df),
        "features": {},
        "prediction": {},
        "alerts": [],
    }

    # Feature-level drift
    feat_stats = baseline.get("feature_stats", {})
    for feat, stats in feat_stats.items():
        if feat not in prod_df.columns:
            log.warning(f"Feature '{feat}' missing from production data!")
            report["alerts"].append({"type": "MISSING_FEATURE", "feature": feat})
            continue

        ref_vals  = np.random.normal(stats["mean"], stats["std"], 5000)  # reconstituted
        prod_vals = prod_df[feat].dropna().values

        psi  = compute_psi(ref_vals, prod_vals)
        ks_stat, ks_p = ks_2samp(ref_vals, prod_vals)
        wass = wasserstein_distance(ref_vals, prod_vals)

        drift_flag = psi >= PSI_THRESHOLD or ks_p < 0.05
        report["features"][feat] = {
            "psi": round(psi, 4),
            "ks_stat": round(float(ks_stat), 4),
            "ks_pvalue": round(float(ks_p), 4),
            "wasserstein": round(float(wass), 4),
            "drift": drift_flag,
        }
        if drift_flag:
            severity = "CRITICAL" if psi >= 0.25 else "WARNING"
            msg = f"[{severity}] Feature '{feat}': PSI={psi:.3f}, KS_p={ks_p:.4f}"
            log.warning(msg)
            report["alerts"].append({
                "type": "FEATURE_DRIFT", "feature": feat,
                "severity": severity, "psi": psi
            })

    # Prediction drift
    if "pred_prob" in prod_df.columns:
        pred_stats = baseline.get("prediction_stats", {})
        prod_probs = prod_df["pred_prob"].values
        baseline_mean = pred_stats.get("mean_prob", 0.5)
        baseline_rate = pred_stats.get("malignant_rate", 0.03)

        prod_rate = float((prod_probs >= 0.5).mean())
        rate_drift = abs(prod_rate - baseline_rate) > 0.05

        report["prediction"] = {
            "prod_mean_prob":     round(float(prod_probs.mean()), 4),
            "baseline_mean_prob": round(baseline_mean, 4),
            "prod_malignant_rate":     round(prod_rate, 4),
            "baseline_malignant_rate": round(baseline_rate, 4),
            "rate_drift": rate_drift,
        }
        if rate_drift:
            msg = (f"[WARNING] Prediction rate drift: "
                   f"prod={prod_rate:.3f} vs baseline={baseline_rate:.3f}")
            log.warning(msg)
            report["alerts"].append({"type": "PREDICTION_DRIFT", "severity": "WARNING"})

    report["has_critical_alert"] = any(
        a.get("severity") == "CRITICAL" for a in report["alerts"]
    )
    return report


# ── Main ──────────────────────────────────────────────────
def main():
    log.info("Starting drift check …")
    try:
        report = run_drift_check()
    except FileNotFoundError as e:
        log.error(str(e))
        return

    # Save JSON report
    date_str  = datetime.utcnow().strftime("%Y%m%d_%H%M")
    out_path  = Path(PROD_LOG_DIR) / f"drift_report_{date_str}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Drift report saved: {out_path}")

    # Summary
    n_drifted = sum(1 for v in report["features"].values() if v.get("drift"))
    n_total   = len(report["features"])
    log.info(f"Features drifted: {n_drifted}/{n_total}")

    if report.get("has_critical_alert"):
        log.critical("⚠️  CRITICAL DRIFT DETECTED — consider retraining!")
    elif report["alerts"]:
        log.warning(f"⚡ {len(report['alerts'])} drift alerts found.")
    else:
        log.info("✅  No significant drift detected.")


if __name__ == "__main__":
    import schedule, time

    log.info(f"Drift monitor starting. Check interval: {CHECK_INTERVAL}h")
    main()  # run immediately on start
    schedule.every(CHECK_INTERVAL).hours.do(main)
    while True:
        schedule.run_pending()
        time.sleep(60)
