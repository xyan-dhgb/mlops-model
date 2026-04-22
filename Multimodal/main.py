"""
main.py — ISIC 2024 Multimodal Training Entry Point
=====================================================
Usage:
    python main.py                              # use default config
    python main.py --config config/train_config.yaml
    python main.py --eval-only --checkpoint final/best_model_isic2024.h5
"""

import argparse
import os
import yaml
import numpy as np

from data_loader.dataloader import build_dataloaders
from models.multimodal_model import build_multimodal_model
from training.train import train_model
from utils.metrics import (
    find_optimal_threshold,
    evaluate_model,
    plot_training_history,
    plot_roc_pr,
)


def load_config(path: str = "config/train_config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main(args):
    cfg = load_config(args.config)
    os.makedirs(cfg["artifacts"]["final_dir"], exist_ok=True)

    # ── 1. Build data ─────────────────────────────────────────────────────────
    print("\n>>> STEP 1: Loading data…")
    train, val, test, meta = build_dataloaders(cfg)

    tabular_shape = (meta["n_tabular_features"],)
    image_shape   = meta["image_shape"]

    print(f"    Tabular features : {tabular_shape[0]}")
    print(f"    Image shape      : {image_shape}")

    # ── 2. Build model ────────────────────────────────────────────────────────
    print("\n>>> STEP 2: Building model…")
    model, backbone = build_multimodal_model(
        tabular_shape=tabular_shape,
        image_shape=image_shape,
        freeze_backbone=True,
        focal_gamma=cfg["loss"].get("focal_gamma", 2.0),
        focal_alpha=cfg["loss"].get("focal_alpha", 0.75),
    )
    model.summary(line_length=100)

    if args.eval_only:
        # ── Eval-only mode ────────────────────────────────────────────────────
        print(f"\n>>> EVAL-ONLY mode — loading weights from {args.checkpoint}")
        model.load_weights(args.checkpoint)
        y_pred_prob = model.predict(
            {"image_input": test["images"], "tabular_input": test["tabular"]},
            verbose=0,
        ).flatten()
        thr, _ = find_optimal_threshold(
            test["labels"], y_pred_prob,
            metric=cfg["evaluation"].get("threshold_strategy", "f1"),
            min_recall=cfg["evaluation"].get("min_recall", 0.60),
        )
        evaluate_model(model, test, threshold=thr,
                       save_path=cfg["artifacts"]["results_path"])
        plot_roc_pr(test["labels"], y_pred_prob, threshold=thr,
                    save_path=os.path.join(cfg["artifacts"]["final_dir"], "roc_pr.png"))
        return

    # ── 3. Train ──────────────────────────────────────────────────────────────
    print("\n>>> STEP 3: Training (2-phase)…")
    model, history1, history2 = train_model(cfg, train, val, model, backbone)

    # ── 4. Threshold tuning ───────────────────────────────────────────────────
    print("\n>>> STEP 4: Threshold tuning on validation set…")
    y_val_prob = model.predict(
        {"image_input": val["images"], "tabular_input": val["tabular"]},
        verbose=0,
    ).flatten()
    threshold, _ = find_optimal_threshold(
        val["labels"], y_val_prob,
        metric=cfg["evaluation"].get("threshold_strategy", "f1"),
        min_recall=cfg["evaluation"].get("min_recall", 0.60),
    )

    # ── 5. Evaluate on test set ───────────────────────────────────────────────
    print("\n>>> STEP 5: Evaluating on test set…")
    results = evaluate_model(
        model, test,
        threshold=threshold,
        save_path=cfg["artifacts"]["results_path"],
    )

    y_test_prob = model.predict(
        {"image_input": test["images"], "tabular_input": test["tabular"]},
        verbose=0,
    ).flatten()

    plot_training_history(
        history1, history2,
        save_path=os.path.join(cfg["artifacts"]["final_dir"], "training_history.png"),
    )
    plot_roc_pr(
        test["labels"], y_test_prob, threshold=threshold,
        save_path=os.path.join(cfg["artifacts"]["final_dir"], "roc_pr.png"),
    )

    print("\n>>> Training complete!")
    print(f"    AUC           : {results['roc_auc']:.4f}")
    print(f"    F1 (Malignant): {results['f1_malignant']:.4f}")
    print(f"    Threshold     : {threshold:.2f}")
    print(f"    Results saved : {cfg['artifacts']['results_path']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ISIC 2024 Multimodal Training")
    parser.add_argument("--config",     default="config/train_config.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--eval-only",  action="store_true",
                        help="Skip training; evaluate a saved checkpoint")
    parser.add_argument("--checkpoint", default="final/best_model_isic2024.h5",
                        help="Checkpoint to load in eval-only mode")
    args = parser.parse_args()
    main(args)
