"""
scripts/run_train.py
Main entry point for training runs — called by CI/CD and manually.

Usage:
    # Default config
    python scripts/run_train.py

    # Custom config
    python scripts/run_train.py --config Multimodal/config/train_config.yaml

    # Override specific params
    python scripts/run_train.py --fold 1 --epochs 50 --batch-size 64

    # Full pipeline (preprocessing + training)
    python scripts/run_train.py --full-pipeline --csv data/meta.csv --image-dir data/images/
"""

import argparse
import logging
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from Multimodal.training.train import train, DEFAULT_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Multimodal Skin Cancer Model")

    # Config file (base)
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config (overrides defaults)")

    # Data args
    parser.add_argument("--csv",       type=str, default=None)
    parser.add_argument("--image-dir", type=str, default=None)
    parser.add_argument("--fold",      type=int, default=None,
                        help="Validation fold index 0-4")

    # Training args
    parser.add_argument("--epochs",     type=int,   default=None)
    parser.add_argument("--batch-size", type=int,   default=None)
    parser.add_argument("--lr",         type=float, default=None)
    parser.add_argument("--device",     type=str,   default=None,
                        choices=["cuda", "cpu"])

    # MLflow
    parser.add_argument("--mlflow-uri",  type=str, default=None)
    parser.add_argument("--run-name",    type=str, default=None)
    parser.add_argument("--experiment",  type=str, default=None)

    # Pipeline mode
    parser.add_argument("--full-pipeline", action="store_true",
                        help="Run preprocessing + training end-to-end")

    return parser.parse_args()


def build_config(args) -> dict:
    """Merge DEFAULT_CONFIG ← YAML ← CLI args (CLI has highest priority)."""
    cfg = DEFAULT_CONFIG.copy()

    # Load YAML
    if args.config:
        path = Path(args.config)
        if not path.exists():
            log.error("Config file not found: %s", args.config)
            sys.exit(1)
        with open(path) as f:
            yaml_cfg = yaml.safe_load(f)
        cfg.update(yaml_cfg)
        log.info("Loaded config from %s", args.config)

    # CLI overrides
    overrides = {
        "csv_path":            args.csv,
        "image_dir":           args.image_dir,
        "fold":                args.fold,
        "num_epochs":          args.epochs,
        "batch_size":          args.batch_size,
        "lr":                  args.lr,
        "device":              args.device,
        "mlflow_tracking_uri": args.mlflow_uri,
        "run_name":            args.run_name,
        "experiment_name":     args.experiment,
    }
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
            log.info("CLI override: %s = %s", k, v)

    return cfg


def main():
    args   = parse_args()
    cfg    = build_config(args)

    log.info("=" * 60)
    log.info("Multimodal Skin Cancer Training")
    log.info("  Experiment : %s", cfg["experiment_name"])
    log.info("  Run name   : %s", cfg["run_name"])
    log.info("  Fold       : %d / 5-fold CV", cfg["fold"])
    log.info("  Epochs     : %d", cfg["num_epochs"])
    log.info("  Batch size : %d", cfg["batch_size"])
    log.info("  Device     : %s", cfg["device"])
    log.info("  MLflow URI : %s", cfg["mlflow_tracking_uri"])
    log.info("=" * 60)

    if args.full_pipeline:
        from src.train import full_pipeline
        full_pipeline(cfg)
    else:
        train(cfg)

    log.info("Done.")


if __name__ == "__main__":
    main()
