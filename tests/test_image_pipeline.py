"""
tests/integration/test_image_pipeline.py
Integration tests: Chạy pipeline thật với MinIO + MLflow thật
Dùng khi có service containers trong CI
"""

import os
import io
import json
import pytest
import numpy as np
import pandas as pd
from PIL import Image
import boto3
import mlflow

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))

from src.preprocessing.image_preprocessing import (
    ImagePreprocessor, MinIOClient, DatasetBuilder,
    LABEL_MAP, IMAGE_SIZE, BUCKET_PROC
)

# ── Skip nếu không có MinIO thật ──────────────────────────
MINIO_AVAILABLE = False
try:
    import requests
    endpoint = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
    r = requests.get(f"{endpoint}/minio/health/live", timeout=3)
    MINIO_AVAILABLE = r.status_code == 200
except Exception:
    pass

skip_no_minio = pytest.mark.skipif(
    not MINIO_AVAILABLE,
    reason="MinIO không available — chạy unit tests thay thế"
)


# ══════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════
@pytest.fixture(scope="module")
def minio_client():
    """MinIOClient thật kết nối tới MinIO test server"""
    client = MinIOClient()
    # Tạo buckets cần thiết
    for bucket in ["raw-data", "processed-data", "mlflow-artifacts"]:
        client.ensure_bucket(bucket)
    return client


@pytest.fixture(scope="module")
def sample_dataset(tmp_path_factory):
    """Tạo dataset nhỏ: 5 ảnh × 6 classes = 30 ảnh"""
    base = tmp_path_factory.mktemp("sample_data")
    img_dir = base / "images"
    img_dir.mkdir()

    rows = []
    img_id = 1000

    for dx_code in LABEL_MAP.keys():
        for i in range(5):
            # Tạo ảnh ngẫu nhiên 300x400
            arr  = np.random.randint(50, 200, (400, 300, 3), dtype=np.uint8)
            name = f"ISIC_{img_id:07d}"
            path = img_dir / f"{name}.jpg"
            Image.fromarray(arr).save(str(path))

            rows.append({
                "image_id":     name,
                "age":          np.random.choice([30, 45, 60, np.nan]),
                "sex":          np.random.choice(["male", "female"]),
                "localization": np.random.choice(["back", "face", "trunk"]),
                "dx":           dx_code,
                "dx_type":      np.random.choice(["histo", "follow_up"]),
            })
            img_id += 1

    metadata = pd.DataFrame(rows)
    csv_path = base / "metadata.csv"
    metadata.to_csv(str(csv_path), index=False)

    return {"img_dir": str(img_dir), "csv_path": str(csv_path), "n": len(rows)}


# ══════════════════════════════════════════════════════════
# INTEGRATION TESTS
# ══════════════════════════════════════════════════════════
@skip_no_minio
class TestImagePreprocessorIntegration:
    def test_process_uploads_to_minio(self, minio_client, tmp_path):
        """Ảnh xử lý xong phải xuất hiện trên MinIO"""
        # Tạo ảnh test
        arr = np.random.randint(50, 200, (300, 400, 3), dtype=np.uint8)
        img_path = tmp_path / "ISIC_INT_001.jpg"
        Image.fromarray(arr).save(str(img_path))

        preprocessor = ImagePreprocessor(minio_client)
        records = preprocessor.process_and_upload(
            img_path=str(img_path),
            label="MEL",
            split="val",
            image_id="ISIC_INT_001",
            augment=False
        )

        assert len(records) == 1
        # Kiểm tra file thực sự tồn tại trên MinIO
        minio_client.client.head_object(
            Bucket=BUCKET_PROC,
            Key=records[0]["s3_key"]
        )


@skip_no_minio
class TestDatasetBuilderIntegration:
    def test_full_pipeline_creates_manifests(self, minio_client, sample_dataset):
        """Full pipeline phải tạo manifests cho cả 3 splits"""
        builder = DatasetBuilder(
            raw_image_dirs=[sample_dataset["img_dir"]],
            metadata_csv=sample_dataset["csv_path"],
            minio=minio_client
        )
        manifest = builder.build(version="0.0.1-test")

        # Kiểm tra manifest có đủ 3 splits
        assert "train" in manifest["splits"]
        assert "val"   in manifest["splits"]
        assert "test"  in manifest["splits"]

        # Kiểm tra count hợp lý
        total_orig = sum(
            manifest["splits"][s]["original_count"]
            for s in ["train", "val", "test"]
        )
        assert total_orig == sample_dataset["n"], \
            f"Expected {sample_dataset['n']} original images, got {total_orig}"

    def test_parquet_files_on_minio(self, minio_client, sample_dataset):
        """Parquet files phải tồn tại trên MinIO sau khi pipeline chạy"""
        for split in ["train", "val", "test"]:
            key = f"metadata/{split}_image_manifest.parquet"
            try:
                minio_client.client.head_object(Bucket=BUCKET_PROC, Key=key)
            except Exception:
                pytest.fail(f"Missing Parquet: {BUCKET_PROC}/{key}")

    def test_manifest_json_on_minio(self, minio_client):
        """JSON manifest phải tồn tại trên MinIO"""
        try:
            response = minio_client.client.get_object(
                Bucket=BUCKET_PROC,
                Key="metadata/dataset_v0.0.1-test.json"
            )
            manifest = json.loads(response["Body"].read())
            assert manifest["version"] == "0.0.1-test"
        except Exception as e:
            pytest.fail(f"Manifest JSON không tồn tại: {e}")

    def test_mlflow_run_created(self, minio_client, sample_dataset):
        """Pipeline phải tạo MLflow run"""
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        client_mlflow = mlflow.tracking.MlflowClient()

        # Tìm experiment
        exp = client_mlflow.get_experiment_by_name("cancer-data-preprocessing")
        if exp is None:
            pytest.skip("MLflow experiment chưa được tạo")

        runs = client_mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            filter_string="tags.`mlflow.runName` LIKE 'preprocessing_%'",
            max_results=1
        )
        assert len(runs) > 0, "Không tìm thấy MLflow run nào"
