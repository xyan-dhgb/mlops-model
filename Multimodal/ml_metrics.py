"""
ml_metrics.py — Custom Prometheus metrics cho MLOps multimodal training.

Cách dùng trong train.py:
    from ml_metrics import MetricsServer, record_epoch_metrics, record_batch

Prometheus sẽ scrape endpoint http://<pod-ip>:8000/metrics.
PodMonitor (ml-train-pod-monitor.yaml) tự động discover pod này qua label metrics=enabled.
"""

import time
import threading
import logging

from prometheus_client import (
    Gauge,
    Histogram,
    Counter,
    start_http_server,
    REGISTRY,
)

logger = logging.getLogger(__name__)

# ── 1. Argo Workflows step progress ────────────────────────────────────────
# Gauge theo dõi epoch hiện tại để biết training đang ở đâu
ml_current_epoch = Gauge(
    "ml_current_epoch",
    "Epoch hiện tại đang train",
    ["model"],                   # isic-multimodal
)

ml_total_epochs = Gauge(
    "ml_total_epochs",
    "Tổng số epoch được cấu hình",
    ["model"],
)

# ── 2. Loss & Accuracy ──────────────────────────────────────────────────────
ml_training_loss = Gauge(
    "ml_training_loss",
    "Training loss cuối epoch (thấp hơn là tốt hơn)",
    ["model", "modality"],       # modality: image | csv | fusion
)

ml_validation_loss = Gauge(
    "ml_validation_loss",
    "Validation loss cuối epoch",
    ["model", "modality"],
)

ml_validation_auc = Gauge(
    "ml_validation_auc",
    "Validation AUC-ROC cuối epoch (cao hơn là tốt hơn)",
    ["model", "modality"],
)

ml_validation_pauc = Gauge(
    "ml_validation_pauc",
    "Partial AUC (tpr_min=0.8) — metric chính của ISIC",
    ["model"],
)

# ── 3. Throughput ───────────────────────────────────────────────────────────
ml_samples_per_sec = Gauge(
    "ml_samples_per_sec",
    "Số samples được xử lý mỗi giây",
    ["stage"],                   # train | val
)

# ── 4. Data Loading Latency (Histogram) ────────────────────────────────────
# Histogram cho phép tính p50/p95/p99 trong Grafana
ml_data_loading_seconds = Histogram(
    "ml_data_loading_seconds",
    "Thời gian load 1 batch dữ liệu từ memmap/disk",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)

ml_training_step_seconds = Histogram(
    "ml_training_step_seconds",
    "Thời gian forward+backward pass 1 batch",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
)

# ── 5. OOM và lỗi ──────────────────────────────────────────────────────────
ml_oom_total = Counter(
    "ml_oom_total",
    "Số lần OOM kill (theo dõi để tuning batch size)",
    ["node", "modality"],
)

ml_epoch_total = Counter(
    "ml_epoch_total",
    "Tổng số epoch đã hoàn thành",
    ["model"],
)


# ── Helper class dùng trong train.py ───────────────────────────────────────

class MetricsServer:
    """Khởi động HTTP server expose /metrics endpoint trên port 8000.

    Gọi MetricsServer.start() một lần khi bắt đầu train.py.
    Thread chạy background, không block training loop.
    """
    _started = False

    @classmethod
    def start(cls, port: int = 8000):
        if cls._started:
            return
        try:
            start_http_server(port)
            cls._started = True
            logger.info(f"[Metrics] Prometheus endpoint: http://0.0.0.0:{port}/metrics")
        except OSError as e:
            # Port đã bị chiếm (ví dụ chạy local test) — log warning, không crash
            logger.warning(f"[Metrics] Không thể start metrics server port {port}: {e}")


def record_epoch_metrics(
    epoch: int,
    total_epochs: int,
    train_loss: float,
    val_loss: float,
    val_auc: float,
    val_pauc: float,
    model_name: str = "isic-multimodal",
    modality: str = "fusion",
):
    """Gọi cuối mỗi epoch để update tất cả loss/accuracy metrics.

    Args:
        epoch: epoch hiện tại (0-indexed)
        total_epochs: tổng số epochs
        train_loss: training loss value
        val_loss: validation loss value
        val_auc: validation AUC-ROC
        val_pauc: partial AUC (tpr_min=0.8), metric chính ISIC challenge
        model_name: tên model (label Prometheus)
        modality: "image", "csv", hoặc "fusion"
    """
    ml_current_epoch.labels(model=model_name).set(epoch + 1)
    ml_total_epochs.labels(model=model_name).set(total_epochs)
    ml_training_loss.labels(model=model_name, modality=modality).set(train_loss)
    ml_validation_loss.labels(model=model_name, modality=modality).set(val_loss)
    ml_validation_auc.labels(model=model_name, modality=modality).set(val_auc)
    ml_validation_pauc.labels(model=model_name).set(val_pauc)
    ml_epoch_total.labels(model=model_name).inc()

    logger.info(
        f"[Metrics] epoch={epoch+1}/{total_epochs} "
        f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
        f"val_auc={val_auc:.4f} val_pauc={val_pauc:.4f}"
    )


class BatchTimer:
    """Context manager để đo thời gian load và train mỗi batch.

    Ví dụ:
        timer = BatchTimer()
        with timer.loading():
            batch = next(data_gen)
        with timer.training():
            loss = model.train_on_batch(batch)
        timer.record_throughput(batch_size, stage="train")
    """

    def loading(self):
        return ml_data_loading_seconds.time()

    def training(self):
        return ml_training_step_seconds.time()

    def record_throughput(self, batch_size: int, elapsed_sec: float, stage: str = "train"):
        if elapsed_sec > 0:
            ml_samples_per_sec.labels(stage=stage).set(batch_size / elapsed_sec)
