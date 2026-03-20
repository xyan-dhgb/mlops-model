"""
conftest.py — Cấu hình pytest và fixtures dùng chung (ISIC 2024)

Thay đổi so với phiên bản ISIC 2019:
  - CLASS_NAMES   : ["benign", "malignant"] (binary)
  - CLASS_PROBS   : [0.97, 0.03] (malignant ~3%)
  - META_DIM      : 9 (thêm 4 đặc trưng TBP)
  - NUM_CLASSES   : 1 (logit đơn, BCEWithLogitsLoss)
  - dummy_labels  : float32 binary (0.0 / 1.0)
  - dummy_metadata_df: có thêm cột patient_id, target, tbp_lv_*
  - tiny_model    : num_classes=1, metadata_input_dim=9
  - criterion     : BinaryFocalLoss thay vì FocalLoss 7-class
  - HDF5 fixtures : dummy hdf5 in-memory cho ISICDataset tests

Fixtures chia thành 7 nhóm:
  1. PYTHONPATH & hằng số dùng chung
  2. Fixtures dữ liệu ảnh
  3. Fixtures dữ liệu bảng (tabular metadata ISIC 2024)
  4. Fixtures mô hình (model, optimizer, criterion)
  5. Fixtures DataLoader in-memory
  6. Fixtures config, đường dẫn, HDF5 tạm thời
  7. Markers tùy chỉnh & pytest hooks

Cách dùng trong test file (không cần import thêm):
    def test_something(dummy_image, dummy_metadata_df, tiny_model):
        ...
"""

import sys
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

# ── PYTHONPATH ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Multimodal.models.multimodal_model import BinaryFocalLoss, MultimodalSkinClassifier
from Multimodal.preprocessing.tabular_preprocessing import (
    CLASS_NAMES,
    SITE_CATEGORIES,
    TARGET_COL,
    TBP_FEATURE_COLS,
)


# =============================================================================
# PHẦN 1 — Hằng số dùng chung (ISIC 2024)
# =============================================================================

BATCH_SIZE  = 4
IMAGE_SIZE  = 224
META_DIM    = 9        # ISIC 2024: 9 chiều (thêm 4 TBP features so với ISIC 2019)
NUM_CLASSES = 1        # binary: logit đơn + BCEWithLogitsLoss
DEVICE      = torch.device("cpu")
SEED        = 42

# ISIC 2024: binary, malignant ~3%
CLASS_NAMES_LOCAL = ["benign", "malignant"]
CLASS_PROBS       = [0.97, 0.03]

# Tập con TBP features dùng để tạo dummy data
TBP_DUMMY_COLS = [
    "tbp_lv_nevi_confidence",
    "tbp_lv_deltaA", "tbp_lv_deltaB",
    "tbp_lv_areaMM2",
    "tbp_lv_norm_border", "tbp_lv_norm_color",
]


# =============================================================================
# PHẦN 2 — Fixtures dữ liệu ảnh
# =============================================================================

@pytest.fixture
def dummy_image() -> np.ndarray:
    """
    Ảnh RGB ngẫu nhiên 512×512 — mô phỏng TBP crop chưa qua resize.
    Dùng cho: TestHairRemoval, TestColorConstancy, TestTransforms.
    """
    rng = np.random.default_rng(SEED)
    return rng.integers(0, 255, (512, 512, 3), dtype=np.uint8)


@pytest.fixture
def dummy_image_small() -> np.ndarray:
    """Ảnh RGB 224×224 đã resize — không cần test bước resize."""
    rng = np.random.default_rng(SEED)
    return rng.integers(0, 255, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)


@pytest.fixture
def dummy_image_tensor() -> torch.Tensor:
    """Tensor ảnh đơn (C, H, W) float32 đã chuẩn hóa ImageNet."""
    torch.manual_seed(SEED)
    return torch.randn(3, IMAGE_SIZE, IMAGE_SIZE)


@pytest.fixture
def dummy_image_batch() -> torch.Tensor:
    """Batch ảnh (BATCH_SIZE, C, H, W) float32."""
    torch.manual_seed(SEED)
    return torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)


# =============================================================================
# PHẦN 3 — Fixtures dữ liệu bảng (ISIC 2024)
# =============================================================================

@pytest.fixture
def dummy_metadata_df() -> pd.DataFrame:
    """
    DataFrame metadata ISIC 2024 tổng hợp — 100 hàng.

    Thay đổi so với ISIC 2019:
      - Cột ID     : isic_id (không phải image_name)
      - Cột nhãn   : target (0/1, không phải diagnosis string)
      - Cột site   : anatom_site_general (bỏ _challenge)
      - Có patient_id: mỗi bệnh nhân 4 ảnh (cho StratifiedGroupKFold)
      - Có tbp_lv_* : các đặc trưng hình học TBP
      - Phân phối  : malignant ~3% (nặng hơn melanoma ~11%)

    Cố ý thêm giá trị null để test clean_metadata.
    """
    rng = np.random.default_rng(SEED)
    n = 100
    data = {
        "isic_id":    [f"ISIC_{i:07d}" for i in range(n)],
        "patient_id": [f"PAT_{i // 4:04d}" for i in range(n)],  # 4 ảnh/bệnh nhân
        "target":     rng.choice([0, 1], n, p=CLASS_PROBS),
        "age_approx": rng.choice([25.0, 40.0, 55.0, 70.0, np.nan], n),
        "sex":        rng.choice(["male", "female", None], n),
        "anatom_site_general": rng.choice(SITE_CATEGORIES + [None], n),
        # TBP features (có null)
        "tbp_lv_nevi_confidence": rng.uniform(0, 1, n),
        "tbp_lv_deltaA":          rng.choice([*rng.normal(0, 5, n - 5), *[np.nan] * 5]),
        "tbp_lv_deltaB":          rng.normal(0, 5, n),
        "tbp_lv_areaMM2":         rng.uniform(1, 50, n),
        "tbp_lv_norm_border":     rng.uniform(0, 1, n),
        "tbp_lv_norm_color":      rng.uniform(0, 1, n),
    }
    return pd.DataFrame(data)


@pytest.fixture
def dummy_metadata_df_large() -> pd.DataFrame:
    """
    DataFrame lớn hơn — 500 hàng, đảm bảo đủ mẫu malignant cho mọi fold.
    Dùng cho test StratifiedGroupKFold không bị lỗi phân phối.
    """
    rng = np.random.default_rng(SEED)
    n = 500
    return pd.DataFrame({
        "isic_id":    [f"ISIC_{i:07d}" for i in range(n)],
        "patient_id": [f"PAT_{i // 4:04d}" for i in range(n)],
        "target":     rng.choice([0, 1], n, p=CLASS_PROBS),
        "age_approx": rng.uniform(20, 80, n),
        "sex":        rng.choice(["male", "female"], n),
        "anatom_site_general": rng.choice(SITE_CATEGORIES, n),
        "tbp_lv_nevi_confidence": rng.uniform(0, 1, n),
        "tbp_lv_deltaA":          rng.normal(0, 5, n),
        "tbp_lv_deltaB":          rng.normal(0, 5, n),
        "tbp_lv_areaMM2":         rng.uniform(1, 50, n),
        "tbp_lv_norm_border":     rng.uniform(0, 1, n),
        "tbp_lv_norm_color":      rng.uniform(0, 1, n),
    })


@pytest.fixture
def dummy_metadata_df_no_nulls(dummy_metadata_df_large) -> pd.DataFrame:
    """
    DataFrame đã sạch — không null, có cột fold (từ create_folds).
    Dùng khi test không cần kiểm tra bước làm sạch.
    """
    from Multimodal.preprocessing.tabular_preprocessing import clean_metadata, create_folds
    df = clean_metadata(dummy_metadata_df_large)
    return create_folds(df)


@pytest.fixture
def dummy_metadata_tensor() -> torch.Tensor:
    """Tensor metadata đã mã hóa (BATCH_SIZE, META_DIM=9) float32."""
    torch.manual_seed(SEED)
    return torch.randn(BATCH_SIZE, META_DIM)


@pytest.fixture
def dummy_labels() -> torch.Tensor:
    """
    Tensor nhãn binary float32 (BATCH_SIZE,) — 0.0 hoặc 1.0.
    ISIC 2024: float32 cho BCEWithLogitsLoss (không phải long).
    """
    torch.manual_seed(SEED)
    return torch.randint(0, 2, (BATCH_SIZE,)).float()


@pytest.fixture
def dummy_batch(dummy_image_batch, dummy_metadata_tensor, dummy_labels):
    """
    Tuple đầy đủ (images, metadata, labels) cho một batch.
    labels là float32 — khác ISIC 2019 dùng long.
    Dùng cho: TestMultimodalModel, TestBinaryFocalLoss,
              TestTrainEpoch, TestValEpoch.
    """
    return dummy_image_batch, dummy_metadata_tensor, dummy_labels


# =============================================================================
# PHẦN 4 — Fixtures mô hình (ISIC 2024)
# =============================================================================

@pytest.fixture(scope="module")
def tiny_model() -> MultimodalSkinClassifier:
    """
    MultimodalSkinClassifier cho ISIC 2024:
      - num_classes=1 (binary logit)
      - metadata_input_dim=9 (9 TBP features)
      - pretrained=False (không tải ImageNet trong CI)
    """
    return MultimodalSkinClassifier(
        num_classes=1,
        metadata_input_dim=META_DIM,
        pretrained=False,
    )


@pytest.fixture(scope="module")
def model() -> MultimodalSkinClassifier:
    """Alias tiny_model ở chế độ eval — dùng cho test_model.py."""
    return MultimodalSkinClassifier(
        num_classes=1,
        metadata_input_dim=META_DIM,
        pretrained=False,
    ).eval()


@pytest.fixture
def optimizer(tiny_model) -> torch.optim.Optimizer:
    """AdamW lr=1e-4 — khởi tạo lại mỗi test."""
    return torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)


@pytest.fixture
def criterion() -> BinaryFocalLoss:
    """
    BinaryFocalLoss(gamma=2.0, alpha=0.25) — không có pos_weight.
    Dùng cho test không cần pos_weight thực tế.
    """
    return BinaryFocalLoss(gamma=2.0, alpha=0.25)


@pytest.fixture
def criterion_with_pos_weight() -> BinaryFocalLoss:
    """
    BinaryFocalLoss với pos_weight=33 — mô phỏng ISIC 2024 thực tế
    (n_neg/n_pos ≈ 97/3 ≈ 33). Dùng để kiểm tra pos_weight có tác dụng.
    """
    pos_weight = torch.tensor(33.0)
    return BinaryFocalLoss(gamma=2.0, pos_weight=pos_weight, alpha=0.25)


# =============================================================================
# PHẦN 5 — Fixtures DataLoader in-memory
# =============================================================================

@pytest.fixture
def tiny_loader() -> DataLoader:
    """
    DataLoader nhỏ — 8 mẫu (2 batch × 4), tensor ngẫu nhiên.
    labels là float32 cho BCEWithLogitsLoss.
    Dùng cho: TestTrainEpoch, TestValEpoch.
    """
    torch.manual_seed(SEED)
    imgs   = torch.randn(8, 3, IMAGE_SIZE, IMAGE_SIZE)
    meta   = torch.randn(8, META_DIM)
    labels = torch.randint(0, 2, (8,)).float()   # float32 binary
    ds = TensorDataset(imgs, meta, labels)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)


@pytest.fixture
def single_sample_loader() -> DataLoader:
    """
    DataLoader 1 mẫu — kiểm tra BatchNorm với batch_size=1 ở eval mode.
    """
    torch.manual_seed(SEED)
    imgs   = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    meta   = torch.randn(1, META_DIM)
    labels = torch.zeros(1)   # benign
    ds = TensorDataset(imgs, meta, labels)
    return DataLoader(ds, batch_size=1, shuffle=False)


@pytest.fixture
def balanced_loader() -> DataLoader:
    """
    DataLoader 32 mẫu với 50% malignant — dùng khi test metric cần
    cả hai lớp trong y_true (AUC-ROC, pAUC không lỗi ValueError).
    """
    torch.manual_seed(SEED)
    imgs   = torch.randn(32, 3, IMAGE_SIZE, IMAGE_SIZE)
    meta   = torch.randn(32, META_DIM)
    # 16 benign + 16 malignant — cân bằng để test metric
    labels = torch.tensor([i % 2 for i in range(32)], dtype=torch.float32)
    ds = TensorDataset(imgs, meta, labels)
    return DataLoader(ds, batch_size=8, shuffle=False)


@pytest.fixture
def imbalanced_loader() -> DataLoader:
    """
    DataLoader mô phỏng tỉ lệ ISIC 2024 (~3% malignant).
    Dùng để test WeightedRandomSampler và pos_weight thực tế.
    """
    torch.manual_seed(SEED)
    n = 64
    imgs   = torch.randn(n, 3, IMAGE_SIZE, IMAGE_SIZE)
    meta   = torch.randn(n, META_DIM)
    # ~3% malignant = 2/64
    labels = torch.zeros(n)
    labels[0] = 1.0
    labels[1] = 1.0
    ds = TensorDataset(imgs, meta, labels)
    return DataLoader(ds, batch_size=8, shuffle=False)


# =============================================================================
# PHẦN 6 — Fixtures config & đường dẫn
# =============================================================================

@pytest.fixture(scope="session")
def project_root() -> Path:
    """Đường dẫn gốc repository."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def config_path() -> Path:
    """Đường dẫn tới train_config.yaml."""
    return PROJECT_ROOT / "Multimodal" / "config" / "train_config.yaml"


@pytest.fixture(scope="session")
def train_config(config_path) -> dict:
    """
    Dict config từ train_config.yaml thật.
    Tự động skip nếu file không tồn tại.
    """
    if not config_path.exists():
        pytest.skip(f"Không tìm thấy train_config.yaml: {config_path}")
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def test_config() -> dict:
    """
    Dict config tối giản cho test — không cần file YAML.
    Nhất quán với DEFAULT_CONFIG trong train.py (ISIC 2024).
    """
    return {
        "experiment_name":      "test_isic2024",
        "run_name":             "test_binary_fold0",
        "mlflow_tracking_uri":  "http://localhost:5000",
        "csv_path":             "Multimodal/data/raw/train-metadata.csv",
        "hdf5_path":            "Multimodal/data/raw/train-image.hdf5",
        "fold":                 0,
        "preprocessor_path":    None,
        "image_size":           IMAGE_SIZE,
        "apply_hair_removal":   False,
        "apply_color_constancy": True,
        "batch_size":           BATCH_SIZE,
        "num_workers":          0,          # tránh lỗi multiprocessing trong CI
        "use_weighted_sampler": True,
        "num_epochs":           2,
        "lr":                   1e-4,
        "weight_decay":         1e-4,
        "gamma_focal":          2.0,
        "focal_alpha":          0.25,
        "scheduler_t_max":      2,
        "grad_clip_norm":       1.0,
        "num_classes":          1,          # binary
        "metadata_input_dim":   META_DIM,   # 9
        "pretrained":           False,
        "freeze_bn":            False,
        "device":               "cpu",
        "use_amp":              False,
        "save_dir":             "/tmp/test_checkpoints",
        "best_metric":          "val/pauc",
        "pauc_min_tpr":         0.80,
        "preprocess_output":    "/tmp/test_preprocessing",
        "hash_manifest":        None,
        "n_folds":              5,
        "seed":                 SEED,
        "generate_xrai":        False,
        "generate_shap":        False,
        "shap_top_k":           3,
        "xrai_fast_mode":       True,
    }


@pytest.fixture
def tmp_save_dir(tmp_path) -> Path:
    """Thư mục tạm để test lưu checkpoint — tự xóa sau test."""
    d = tmp_path / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def tmp_preprocessor_path(tmp_path) -> str:
    """Đường dẫn tạm để test save/load MetadataPreprocessor."""
    return str(tmp_path / "metadata_preprocessor.pkl")


@pytest.fixture
def dummy_hdf5_path(tmp_path) -> str:
    """
    File HDF5 tạm thời chứa ảnh JPEG ngẫu nhiên.
    Dùng để test ISICDataset mà không cần dataset thật.

    Cấu trúc: /ISIC_0000000, /ISIC_0000001, ...
    """
    try:
        import h5py
        import cv2

        hdf5_file = tmp_path / "test_images.hdf5"
        rng = np.random.default_rng(SEED)

        with h5py.File(str(hdf5_file), "w") as f:
            for i in range(20):
                isic_id = f"ISIC_{i:07d}"
                # Tạo ảnh ngẫu nhiên và encode sang JPEG bytes
                img_arr = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
                _, buf = cv2.imencode(".jpg", img_arr)
                f.create_dataset(isic_id, data=buf.tobytes())

        return str(hdf5_file)

    except ImportError:
        pytest.skip("h5py hoặc opencv không được cài đặt — bỏ qua HDF5 test")


# =============================================================================
# PHẦN 7 — Markers tùy chỉnh & pytest hooks
# =============================================================================

def pytest_configure(config):
    """
    Đăng ký custom markers.

    CLI examples:
        pytest -m unit             # chỉ unit tests
        pytest -m "not slow"       # bỏ qua test chậm
        pytest -m "not integration"
        pytest -m gpu
    """
    config.addinivalue_line(
        "markers",
        "slow: test chạy lâu (>5s) — bỏ qua trong CI nhanh với -m 'not slow'",
    )
    config.addinivalue_line(
        "markers",
        "unit: kiểm thử đơn vị — không cần GPU, không cần file thật",
    )
    config.addinivalue_line(
        "markers",
        "integration: kiểm thử tích hợp — cần HDF5, CSV thật hoặc MLflow server",
    )
    config.addinivalue_line(
        "markers",
        "gpu: test yêu cầu CUDA GPU — tự động skip nếu không có GPU",
    )


def pytest_collection_modifyitems(config, items):
    """Tự động skip test @gpu nếu CUDA không khả dụng."""
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(
            reason="CUDA không khả dụng — bỏ qua GPU test"
        )
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


def pytest_runtest_setup(item):
    """Đặt seed trước mỗi test để đảm bảo tính tái tạo."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
