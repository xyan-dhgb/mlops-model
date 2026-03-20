"""
conftest.py — Cấu hình pytest và fixtures dùng chung cho toàn bộ test suite

Fixtures được chia thành 7 nhóm:
  1. PYTHONPATH & cấu hình pytest
  2. Hằng số dùng chung (tránh khai báo trùng lặp giữa các file test)
  3. Fixtures dữ liệu ảnh
  4. Fixtures dữ liệu bảng (tabular metadata)
  5. Fixtures mô hình (model, optimizer, criterion)
  6. Fixtures DataLoader in-memory (không cần ảnh thật)
  7. Fixtures config, đường dẫn & markers tùy chỉnh

Scope:
  - session  : tạo một lần cho toàn bộ test session (nặng — model weights)
  - module   : tạo lại mỗi test module
  - function : tạo lại mỗi test function (mặc định — fixture sạch)

Cách dùng trong test file (không cần import thêm):
    def test_something(dummy_image, dummy_metadata_df, tiny_model):
        ...
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

# ── PYTHONPATH: đảm bảo import Multimodal.* hoạt động từ mọi vị trí ─────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Multimodal.models.multimodal_model import FocalLoss, MultimodalSkinClassifier
from Multimodal.preprocessing.tabular_preprocessing import CLASS_NAMES, SITE_CATEGORIES


# =============================================================================
# PHẦN 1 — Hằng số dùng chung
# Định nghĩa một lần tại đây, tránh khai báo trùng trong từng test file
# =============================================================================

BATCH_SIZE  = 4
IMAGE_SIZE  = 224
META_DIM    = 5        # [age_norm, is_male, high_risk_site, age_bucket, site_encoded]
NUM_CLASSES = 7        # MEL, NV, BCC, AKIEC, BKL, DF, VASC
DEVICE      = torch.device("cpu")   # CI không yêu cầu GPU
SEED        = 42

# Xác suất phân phối lớp khớp với ISIC 2019 thực tế
CLASS_PROBS = [0.11, 0.67, 0.05, 0.03, 0.11, 0.01, 0.02]


# =============================================================================
# PHẦN 2 — Fixtures dữ liệu ảnh
# =============================================================================

@pytest.fixture
def dummy_image() -> np.ndarray:
    """
    Ảnh RGB ngẫu nhiên 512x512 mô phỏng ảnh dermoscopy thô.
    Có cố ý dùng kích thước lớn để test bước resize trong transforms.
    Dùng cho: TestHairRemoval, TestColorConstancy, TestTransforms.
    """
    rng = np.random.default_rng(SEED)
    return rng.integers(0, 255, (512, 512, 3), dtype=np.uint8)


@pytest.fixture
def dummy_image_small() -> np.ndarray:
    """
    Ảnh RGB 224x224 đã qua resize — dùng khi không cần test bước resize.
    """
    rng = np.random.default_rng(SEED)
    return rng.integers(0, 255, (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)


@pytest.fixture
def dummy_image_tensor() -> torch.Tensor:
    """
    Tensor ảnh đơn lẻ (C, H, W) float32 — đầu vào trực tiếp cho model.
    Giá trị nằm ngoài [0,1] do ImageNet normalization.
    """
    torch.manual_seed(SEED)
    return torch.randn(3, IMAGE_SIZE, IMAGE_SIZE)


@pytest.fixture
def dummy_image_batch() -> torch.Tensor:
    """
    Batch ảnh (BATCH_SIZE, C, H, W) float32 — dùng cho test forward pass.
    """
    torch.manual_seed(SEED)
    return torch.randn(BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)


# =============================================================================
# PHẦN 3 — Fixtures dữ liệu bảng (tabular metadata)
# =============================================================================

@pytest.fixture
def dummy_metadata_df() -> pd.DataFrame:
    """
    DataFrame metadata ISIC tổng hợp — 100 hàng, phân phối lớp thực tế.
    Có cố ý thêm giá trị null để kiểm thử bước clean_metadata.

    Cột: image_name, age_approx, sex, anatom_site_general_challenge, diagnosis
    Dùng cho: TestCleanMetadata, TestEngineerFeatures, TestCreateFolds,
              TestMetadataPreprocessor, TestClassWeights.
    """
    rng = np.random.default_rng(SEED)
    return pd.DataFrame({
        "image_name": [f"ISIC_{i:07d}" for i in range(100)],
        "age_approx": rng.choice([25.0, 40.0, 55.0, 70.0, np.nan], 100),
        "sex": rng.choice(["male", "female", None], 100),
        "anatom_site_general_challenge": rng.choice(
            SITE_CATEGORIES + [None], 100
        ),
        "diagnosis": rng.choice(CLASS_NAMES, 100, p=CLASS_PROBS),
    })


@pytest.fixture
def dummy_metadata_df_large() -> pd.DataFrame:
    """
    DataFrame metadata lớn hơn — 500 hàng, đảm bảo đủ mẫu mỗi lớp.
    Dùng cho test cần 5-fold stratification không bị lỗi phân phối.
    """
    rng = np.random.default_rng(SEED)
    return pd.DataFrame({
        "image_name": [f"ISIC_{i:07d}" for i in range(500)],
        "age_approx": rng.uniform(20, 80, 500),
        "sex": rng.choice(["male", "female"], 500),
        "anatom_site_general_challenge": rng.choice(SITE_CATEGORIES, 500),
        "diagnosis": rng.choice(CLASS_NAMES, 500, p=CLASS_PROBS),
    })


@pytest.fixture
def dummy_metadata_df_no_nulls() -> pd.DataFrame:
    """
    DataFrame đã sạch — không có giá trị null, nhãn hợp lệ, có cột label.
    Dùng khi test không cần kiểm tra bước làm sạch dữ liệu.
    """
    rng = np.random.default_rng(SEED)
    n = 100
    labels = rng.choice(range(NUM_CLASSES), n, p=CLASS_PROBS)
    return pd.DataFrame({
        "image_name": [f"ISIC_{i:07d}" for i in range(n)],
        "age_approx": rng.uniform(20, 80, n),
        "sex": rng.choice(["male", "female"], n),
        "anatom_site_general_challenge": rng.choice(SITE_CATEGORIES, n),
        "diagnosis": [CLASS_NAMES[l] for l in labels],
        "label": labels,
    })


@pytest.fixture
def dummy_metadata_tensor() -> torch.Tensor:
    """
    Tensor metadata đã mã hóa (BATCH_SIZE, META_DIM) float32.
    Dùng trực tiếp cho test forward pass model.
    """
    torch.manual_seed(SEED)
    return torch.randn(BATCH_SIZE, META_DIM)


@pytest.fixture
def dummy_labels() -> torch.Tensor:
    """
    Tensor nhãn lớp ngẫu nhiên (BATCH_SIZE,) long.
    Dùng cho FocalLoss và tính metrics.
    """
    torch.manual_seed(SEED)
    return torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))


@pytest.fixture
def dummy_batch(dummy_image_batch, dummy_metadata_tensor, dummy_labels):
    """
    Tuple đầy đủ (images, metadata, labels) cho một batch huấn luyện.
    Dùng cho: TestMultimodalModel, TestFocalLoss,
              TestTrainEpoch, TestValEpoch.
    """
    return dummy_image_batch, dummy_metadata_tensor, dummy_labels


# =============================================================================
# PHẦN 4 — Fixtures mô hình
# =============================================================================

@pytest.fixture(scope="module")
def tiny_model() -> MultimodalSkinClassifier:
    """
    MultimodalSkinClassifier nhỏ — pretrained=False để không tải
    ImageNet weights trong CI (tiết kiệm băng thông và thời gian).

    Scope=module: khởi tạo một lần, dùng chung trong toàn bộ test module.
    Lưu ý: mỗi test nên gọi model.train() hoặc model.eval() trước khi dùng.
    """
    return MultimodalSkinClassifier(
        num_classes=NUM_CLASSES,
        metadata_input_dim=META_DIM,
        pretrained=False,
    )


@pytest.fixture(scope="module")
def model() -> MultimodalSkinClassifier:
    """
    Alias tiny_model ở chế độ eval — dùng cho các test trong test_model.py.
    Scope=module để nhất quán với cách dùng trong test file cũ.
    """
    return MultimodalSkinClassifier(
        num_classes=NUM_CLASSES,
        metadata_input_dim=META_DIM,
        pretrained=False,
    ).eval()


@pytest.fixture
def optimizer(tiny_model) -> torch.optim.Optimizer:
    """
    AdamW optimizer với lr=1e-4 — khởi tạo lại mỗi test để tránh
    trạng thái optimizer ảnh hưởng giữa các test case.
    """
    return torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)


@pytest.fixture
def criterion() -> FocalLoss:
    """
    FocalLoss(gamma=2.0) — khớp với giá trị mặc định trong train_config.yaml.
    """
    return FocalLoss(gamma=2.0)


@pytest.fixture
def criterion_with_weights() -> FocalLoss:
    """
    FocalLoss với class weights — melanoma (idx=0) có trọng số cao nhất.
    Dùng để kiểm tra class weights có tác động đến giá trị loss.
    """
    # Trọng số nghịch đảo tần suất: MEL(3.0) > BCC(2.0) > NV(0.5) ...
    weights = torch.tensor([3.0, 0.5, 2.0, 4.0, 2.0, 5.0, 5.0])
    return FocalLoss(alpha=weights, gamma=2.0)


# =============================================================================
# PHẦN 5 — Fixtures DataLoader in-memory
# Không cần CSV hay ảnh thật — dùng tensor ngẫu nhiên
# =============================================================================

@pytest.fixture
def tiny_loader() -> DataLoader:
    """
    DataLoader nhỏ — 8 mẫu (2 batch × 4), toàn bộ là tensor ngẫu nhiên.
    Dùng cho: TestTrainEpoch, TestValEpoch — không cần file thật.
    """
    torch.manual_seed(SEED)
    imgs   = torch.randn(8, 3, IMAGE_SIZE, IMAGE_SIZE)
    meta   = torch.randn(8, META_DIM)
    labels = torch.randint(0, NUM_CLASSES, (8,))
    ds = TensorDataset(imgs, meta, labels)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)


@pytest.fixture
def single_sample_loader() -> DataLoader:
    """
    DataLoader với đúng 1 mẫu — kiểm tra trường hợp biên (edge case)
    của BatchNorm khi batch_size=1 ở chế độ eval.
    """
    torch.manual_seed(SEED)
    imgs   = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    meta   = torch.randn(1, META_DIM)
    labels = torch.randint(0, NUM_CLASSES, (1,))
    ds = TensorDataset(imgs, meta, labels)
    return DataLoader(ds, batch_size=1, shuffle=False)


@pytest.fixture
def larger_loader() -> DataLoader:
    """
    DataLoader 32 mẫu — dùng khi test cần đủ mẫu cho AUC-ROC
    (sklearn yêu cầu ít nhất 1 mẫu mỗi lớp trong y_true).
    """
    torch.manual_seed(SEED)
    imgs   = torch.randn(32, 3, IMAGE_SIZE, IMAGE_SIZE)
    meta   = torch.randn(32, META_DIM)
    # Tạo nhãn có đủ 7 lớp để AUC-ROC không lỗi
    labels = torch.tensor(
        [i % NUM_CLASSES for i in range(32)], dtype=torch.long
    )
    ds = TensorDataset(imgs, meta, labels)
    return DataLoader(ds, batch_size=8, shuffle=False)


# =============================================================================
# PHẦN 6 — Fixtures config & đường dẫn
# =============================================================================

@pytest.fixture(scope="session")
def project_root() -> Path:
    """Đường dẫn gốc của repository."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def config_path() -> Path:
    """Đường dẫn tới train_config.yaml."""
    return PROJECT_ROOT / "Multimodal" / "config" / "train_config.yaml"


@pytest.fixture(scope="session")
def train_config(config_path) -> dict:
    """
    Dict config đọc từ train_config.yaml thật trên disk.
    Dùng để kiểm tra tính nhất quán giữa config và code
    (ví dụ: num_classes trong config khớp với NUM_CLASSES trong model).
    Tự động skip nếu file chưa tồn tại.
    """
    if not config_path.exists():
        pytest.skip(f"Không tìm thấy train_config.yaml: {config_path}")
    import yaml
    with open(config_path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def test_config() -> dict:
    """
    Dict config tối giản cho test — không cần file YAML, không cần file thật.
    Tất cả giá trị nhất quán với DEFAULT_CONFIG trong train.py.

    Điểm khác so với config thật:
      - num_epochs=2        (chạy nhanh)
      - num_workers=0       (tránh lỗi multiprocessing trong CI)
      - pretrained=False    (không tải ImageNet weights)
      - use_amp=False       (AMP không hỗ trợ trên CPU)
      - device="cpu"        (CI không có GPU)
    """
    return {
        "experiment_name":      "test_experiment",
        "run_name":             "test_run_fold0",
        "mlflow_tracking_uri":  "http://localhost:5000",
        "csv_path":             "Multimodal/data/raw/ISIC_2019_Training_Metadata.csv",
        "image_dir":            "Multimodal/data/raw/ISIC_2019_Training_Input",
        "fold":                 0,
        "preprocessor_path":    None,
        "image_size":           IMAGE_SIZE,
        "apply_hair_removal":   True,
        "apply_color_constancy": True,
        "batch_size":           BATCH_SIZE,
        "num_workers":          0,
        "use_weighted_sampler": True,
        "num_epochs":           2,
        "lr":                   1e-4,
        "weight_decay":         1e-4,
        "gamma_focal":          2.0,
        "scheduler_t_max":      2,
        "grad_clip_norm":       1.0,
        "num_classes":          NUM_CLASSES,
        "metadata_input_dim":   META_DIM,
        "pretrained":           False,
        "freeze_bn":            False,
        "device":               "cpu",
        "use_amp":              False,
        "save_dir":             "/tmp/test_checkpoints",
        "best_metric":          "val/auc_roc_macro",
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
    """
    Thư mục tạm thời để test lưu checkpoint và artifact.
    Tự động xóa sau mỗi test (pytest quản lý tmp_path).
    """
    save_dir = tmp_path / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


@pytest.fixture
def tmp_preprocessor_path(tmp_path) -> str:
    """
    Đường dẫn tạm thời để test save/load MetadataPreprocessor.
    Dùng cho: TestMetadataPreprocessor.test_save_load_roundtrip
    """
    return str(tmp_path / "metadata_preprocessor.pkl")


# =============================================================================
# PHẦN 7 — Markers tùy chỉnh & pytest hooks
# =============================================================================

def pytest_configure(config):
    """
    Đăng ký custom markers — phân loại test để chạy có chọn lọc.

    CLI examples:
        pytest -m unit                    # chỉ chạy unit tests
        pytest -m "not slow"              # bỏ qua test chạy lâu
        pytest -m "not integration"       # bỏ qua test cần file thật
        pytest -m gpu                     # chỉ chạy test cần GPU
    """
    config.addinivalue_line(
        "markers",
        "slow: test chạy lâu (>5 giây) — bỏ qua trong CI nhanh với -m 'not slow'",
    )
    config.addinivalue_line(
        "markers",
        "unit: kiểm thử đơn vị — không cần GPU, không cần file thật",
    )
    config.addinivalue_line(
        "markers",
        "integration: kiểm thử tích hợp — cần CSV, ảnh thật hoặc MLflow server",
    )
    config.addinivalue_line(
        "markers",
        "gpu: test yêu cầu CUDA GPU — tự động skip nếu không có GPU",
    )


def pytest_collection_modifyitems(config, items):
    """
    Hook tự động: skip test có marker 'gpu' nếu CUDA không khả dụng.
    Không cần thêm @pytest.mark.skipif thủ công vào từng test.
    """
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(
            reason="CUDA không khả dụng — bỏ qua test yêu cầu GPU"
        )
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


def pytest_runtest_setup(item):
    """
    Hook trước mỗi test: đặt seed ngẫu nhiên để đảm bảo tính tái tạo.
    Tránh test sau bị ảnh hưởng bởi trạng thái random của test trước.
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)
