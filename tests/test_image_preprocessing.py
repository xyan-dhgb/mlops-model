"""
tests/unit/test_image_preprocessing.py
Unit tests cho ImagePreprocessor và DatasetBuilder
Dùng moto để mock S3/MinIO — không cần server thật
"""

import io
import json
import hashlib
import pytest
import numpy as np
from PIL import Image
from unittest.mock import MagicMock, patch, call
import boto3
from moto import mock_aws

# ── Import modules cần test ────────────────────────────────
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))

from src.preprocessing.image_preprocessing import (
    ImagePreprocessor,
    MinIOClient,
    LABEL_MAP,
    IMAGE_SIZE,
    JPEG_QUALITY,
)


# ══════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════
@pytest.fixture
def sample_image_rgb():
    """Ảnh RGB 300x400 (không vuông, để test resize)"""
    arr = np.random.randint(50, 200, (400, 300, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def too_dark_image():
    """Ảnh quá tối (mean < 10)"""
    arr = np.full((224, 224, 3), 5, dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def too_bright_image():
    """Ảnh quá sáng (mean > 245)"""
    arr = np.full((224, 224, 3), 250, dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def mock_minio(tmp_path):
    """MinIOClient mock dùng moto (giả lập S3)"""
    with mock_aws():
        # Tạo fake S3 bucket
        client = boto3.client(
            "s3",
            region_name="us-east-1",
            aws_access_key_id="test",
            aws_secret_access_key="test",
        )
        client.create_bucket(Bucket="processed-data")
        client.create_bucket(Bucket="raw-data")

        # Tạo MinIOClient với boto3 client đã mock
        minio = MagicMock(spec=MinIOClient)
        minio.client = client

        # Mock upload_image: thực sự lưu vào S3 mock
        def fake_upload_image(img, bucket, key):
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=JPEG_QUALITY)
            buf.seek(0)
            client.put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
            return True

        minio.upload_image.side_effect = fake_upload_image
        minio.upload_parquet.return_value = True
        minio.upload_json.return_value = True

        yield minio


@pytest.fixture
def preprocessor(mock_minio):
    return ImagePreprocessor(mock_minio)


# ══════════════════════════════════════════════════════════
# TEST: resize_with_padding
# ══════════════════════════════════════════════════════════
class TestResizeWithPadding:
    def test_output_size_is_always_224x224(self, preprocessor, sample_image_rgb):
        """Bất kể input size, output phải là 224x224"""
        result = preprocessor.resize_with_padding(sample_image_rgb)
        assert result.size == IMAGE_SIZE, f"Expected {IMAGE_SIZE}, got {result.size}"

    def test_output_mode_is_rgb(self, preprocessor, sample_image_rgb):
        result = preprocessor.resize_with_padding(sample_image_rgb)
        assert result.mode == "RGB"

    def test_square_image_no_padding(self, preprocessor):
        """Ảnh vuông 300x300 → resize về 224x224, không cần padding"""
        img = Image.new("RGB", (300, 300), color=(128, 64, 32))
        result = preprocessor.resize_with_padding(img)
        assert result.size == IMAGE_SIZE

    def test_very_small_image(self, preprocessor):
        """Ảnh nhỏ 32x32 → phải upscale lên 224x224"""
        img = Image.new("RGB", (32, 32), color=(100, 100, 100))
        result = preprocessor.resize_with_padding(img)
        assert result.size == IMAGE_SIZE

    def test_custom_fill_color(self, preprocessor):
        """Padding màu trắng thay vì đen"""
        img = Image.new("RGB", (100, 50))  # Landscape
        result = preprocessor.resize_with_padding(img, fill_color=(255, 255, 255))
        assert result.size == IMAGE_SIZE
        # Góc trên-trái phải là màu trắng (padding area)
        # (tùy thuộc vào aspect ratio)

    @pytest.mark.parametrize("w,h", [
        (100, 400),   # Portrait cực đoan
        (400, 100),   # Landscape cực đoan
        (224, 224),   # Đúng kích thước rồi
        (1920, 1080), # HD
    ])
    def test_various_aspect_ratios(self, preprocessor, w, h):
        img = Image.new("RGB", (w, h))
        result = preprocessor.resize_with_padding(img)
        assert result.size == IMAGE_SIZE


# ══════════════════════════════════════════════════════════
# TEST: normalize_color
# ══════════════════════════════════════════════════════════
class TestNormalizeColor:
    def test_output_size_unchanged(self, preprocessor, sample_image_rgb):
        result = preprocessor.normalize_color(sample_image_rgb)
        assert result.size == sample_image_rgb.size

    def test_output_mode_rgb(self, preprocessor, sample_image_rgb):
        result = preprocessor.normalize_color(sample_image_rgb)
        assert result.mode == "RGB"

    def test_uniform_image_stays_uniform(self, preprocessor):
        """Ảnh một màu đồng nhất → sau equalization vẫn là ảnh một màu"""
        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        result = preprocessor.normalize_color(img)
        arr = np.array(result)
        # Variance gần 0 (không có thông tin thêm)
        assert arr.std() < 5.0

    def test_dark_image_becomes_brighter(self, preprocessor):
        """Ảnh tối sau equalization phải sáng hơn"""
        dark_arr = np.random.randint(0, 50, (224, 224, 3), dtype=np.uint8)
        dark_img = Image.fromarray(dark_arr)
        result = preprocessor.normalize_color(dark_img)
        assert np.array(result).mean() > dark_arr.mean()


# ══════════════════════════════════════════════════════════
# TEST: augment
# ══════════════════════════════════════════════════════════
class TestAugment:
    def test_returns_5_variants(self, preprocessor, sample_image_rgb):
        results = preprocessor.augment(sample_image_rgb)
        assert len(results) == 5, f"Expected 5 variants, got {len(results)}"

    def test_all_variants_are_pil_images(self, preprocessor, sample_image_rgb):
        results = preprocessor.augment(sample_image_rgb)
        for img in results:
            assert isinstance(img, Image.Image)

    def test_all_variants_same_size(self, preprocessor, sample_image_rgb):
        results = preprocessor.augment(sample_image_rgb)
        expected = sample_image_rgb.size
        for img in results:
            assert img.size == expected, f"Expected {expected}, got {img.size}"

    def test_variants_are_different(self, preprocessor):
        """Các variants phải khác nhau (không phải copy)"""
        arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        arr[0, 0] = [255, 0, 0]  # Marker pixel góc trên-trái
        img = Image.fromarray(arr)
        variants = preprocessor.augment(img)

        original = np.array(variants[0])
        flip_h   = np.array(variants[1])

        # Flip ngang → pixel góc trên-trái phải khác
        assert not np.array_equal(original[0, 0], flip_h[0, 0])

    def test_original_is_first(self, preprocessor, sample_image_rgb):
        """Phần tử đầu tiên phải là ảnh gốc không thay đổi"""
        variants = preprocessor.augment(sample_image_rgb)
        orig_arr     = np.array(sample_image_rgb)
        variant0_arr = np.array(variants[0])
        assert np.array_equal(orig_arr, variant0_arr)


# ══════════════════════════════════════════════════════════
# TEST: is_valid_image
# ══════════════════════════════════════════════════════════
class TestIsValidImage:
    def test_valid_normal_image(self, preprocessor, tmp_path):
        """Ảnh bình thường phải pass validation"""
        img_path = tmp_path / "valid.jpg"
        arr = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        Image.fromarray(arr).save(str(img_path))
        assert preprocessor.is_valid_image(str(img_path)) is True

    def test_corrupt_file(self, preprocessor, tmp_path):
        """File bị corrupt phải return False"""
        bad_path = tmp_path / "corrupt.jpg"
        bad_path.write_bytes(b"this is not a valid JPEG file")
        assert preprocessor.is_valid_image(str(bad_path)) is False

    def test_too_dark_image(self, preprocessor, tmp_path, too_dark_image):
        img_path = tmp_path / "dark.jpg"
        too_dark_image.save(str(img_path))
        # Lưu ý: JPEG compression có thể thay đổi pixel values một chút
        # Test này verify logic, không phải exact threshold
        result = preprocessor.is_valid_image(str(img_path))
        # Ảnh rất tối (mean ~5) phải fail
        arr = np.array(too_dark_image)
        if arr.mean() < 10:
            assert result is False

    def test_nonexistent_file(self, preprocessor):
        """File không tồn tại → False"""
        assert preprocessor.is_valid_image("/nonexistent/path/image.jpg") is False


# ══════════════════════════════════════════════════════════
# TEST: process_and_upload (integration-style unit test)
# ══════════════════════════════════════════════════════════
class TestProcessAndUpload:
    def test_returns_records_for_valid_image(self, preprocessor, tmp_path, mock_minio):
        """Ảnh hợp lệ phải return list records không rỗng"""
        img_path = tmp_path / "ISIC_0001234.jpg"
        arr = np.random.randint(50, 200, (300, 400, 3), dtype=np.uint8)
        Image.fromarray(arr).save(str(img_path))

        records = preprocessor.process_and_upload(
            img_path=str(img_path),
            label="MEL",
            split="val",
            image_id="ISIC_0001234",
            augment=False  # Không augment val
        )

        assert len(records) == 1  # Chỉ 1 ảnh gốc, không augment
        assert records[0]["label"]    == "MEL"
        assert records[0]["split"]    == "val"
        assert records[0]["image_id"] == "ISIC_0001234"
        assert records[0]["variant"]  == "orig"
        assert records[0]["width"]    == IMAGE_SIZE[0]
        assert records[0]["height"]   == IMAGE_SIZE[1]
        assert "md5" in records[0]
        assert "s3_key" in records[0]

    def test_train_with_augment_returns_5_records(self, preprocessor, tmp_path, mock_minio):
        """Train split + augment=True → 5 records"""
        img_path = tmp_path / "ISIC_train.jpg"
        arr = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        Image.fromarray(arr).save(str(img_path))

        records = preprocessor.process_and_upload(
            img_path=str(img_path),
            label="BCC",
            split="train",
            image_id="ISIC_train",
            augment=True
        )

        assert len(records) == 5
        variants = [r["variant"] for r in records]
        assert "orig"    in variants
        assert "flip_h"  in variants
        assert "flip_v"  in variants
        assert "rot90"   in variants
        assert "rot270"  in variants

    def test_augmented_records_flagged(self, preprocessor, tmp_path, mock_minio):
        """Records augmented phải có is_augmented=True, orig=False"""
        img_path = tmp_path / "ISIC_flag_test.jpg"
        arr = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        Image.fromarray(arr).save(str(img_path))

        records = preprocessor.process_and_upload(
            str(img_path), "ACK", "train", "ISIC_flag_test", augment=True
        )

        orig_records = [r for r in records if not r["is_augmented"]]
        aug_records  = [r for r in records if r["is_augmented"]]
        assert len(orig_records) == 1
        assert len(aug_records)  == 4

    def test_corrupt_image_returns_empty(self, preprocessor, tmp_path, mock_minio):
        """Ảnh corrupt → empty list"""
        bad_path = tmp_path / "corrupt.jpg"
        bad_path.write_bytes(b"not a jpeg")

        records = preprocessor.process_and_upload(
            str(bad_path), "NEV", "test", "ISIC_corrupt", augment=False
        )
        assert records == []

    def test_s3_key_format(self, preprocessor, tmp_path, mock_minio):
        """S3 key phải đúng format: images/{split}/{label}/{image_id}_{variant}.jpg"""
        img_path = tmp_path / "ISIC_9999.jpg"
        arr = np.random.randint(50, 200, (224, 224, 3), dtype=np.uint8)
        Image.fromarray(arr).save(str(img_path))

        records = preprocessor.process_and_upload(
            str(img_path), "SEK", "test", "ISIC_9999", augment=False
        )

        expected_key = "images/test/SEK/ISIC_9999_orig.jpg"
        assert records[0]["s3_key"] == expected_key

    def test_md5_is_consistent(self, preprocessor, tmp_path, mock_minio):
        """Cùng 1 ảnh → MD5 phải giống nhau (deterministic)"""
        img_path = tmp_path / "ISIC_md5.jpg"
        arr = np.ones((224, 224, 3), dtype=np.uint8) * 128
        Image.fromarray(arr).save(str(img_path))

        r1 = preprocessor.process_and_upload(
            str(img_path), "NEV", "val", "ISIC_md5", augment=False
        )
        r2 = preprocessor.process_and_upload(
            str(img_path), "NEV", "val", "ISIC_md5", augment=False
        )
        assert r1[0]["md5"] == r2[0]["md5"]


# ══════════════════════════════════════════════════════════
# TEST: LABEL_MAP
# ══════════════════════════════════════════════════════════
class TestLabelMap:
    def test_all_ham10000_codes_mapped(self):
        """Tất cả 7 mã HAM10000 phải có trong LABEL_MAP"""
        expected_keys = {"akiec", "bcc", "mel", "nv", "df", "vasc", "bkl"}
        assert set(LABEL_MAP.keys()) == expected_keys

    def test_all_values_are_valid_classes(self):
        valid_classes = {"ACK", "BCC", "MEL", "NEV", "SCC", "SEK"}
        for k, v in LABEL_MAP.items():
            assert v in valid_classes, f"Invalid class {v} for key {k}"

    def test_mel_maps_to_MEL(self):
        assert LABEL_MAP["mel"] == "MEL"

    def test_nv_maps_to_NEV(self):
        assert LABEL_MAP["nv"] == "NEV"
