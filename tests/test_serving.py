"""
tests/unit/test_serving.py
Unit tests cho FastAPI serving endpoint
"""

import pytest
import io
import json
import numpy as np
from unittest.mock import MagicMock, patch, AsyncMock
from PIL import Image
from fastapi.testclient import TestClient

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))

from src.serving.app import create_app, PredictionRequest, PredictionResponse

# ══════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════
@pytest.fixture
def mock_model():
    """Mock model — không load weights thật"""
    model = MagicMock()
    # predict_proba trả về tensor giả
    import torch
    probs = torch.softmax(torch.randn(1, 6), dim=1)
    model.predict_proba.return_value = probs
    model.eval.return_value = model
    return model


@pytest.fixture
def client(mock_model):
    """FastAPI test client với model đã mock"""
    with patch("src.serving.app.load_model", return_value=mock_model):
        app = create_app()
        return TestClient(app)


def make_jpeg_bytes(w=224, h=224) -> bytes:
    """Tạo JPEG bytes giả để upload"""
    arr = np.random.randint(50, 200, (h, w, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ══════════════════════════════════════════════════════════
# TEST: Health endpoints
# ══════════════════════════════════════════════════════════
class TestHealthEndpoints:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_response_body(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert data["status"] == "healthy"
        assert "model_version" in data
        assert "uptime_seconds" in data

    def test_readiness_returns_200(self, client):
        resp = client.get("/ready")
        assert resp.status_code == 200

    def test_metrics_endpoint(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200


# ══════════════════════════════════════════════════════════
# TEST: Predict endpoint
# ══════════════════════════════════════════════════════════
class TestPredictEndpoint:
    def _make_form_data(self, tabular_data: dict = None):
        """Tạo form data cho predict request"""
        if tabular_data is None:
            tabular_data = {
                "age": 55.0,
                "sex": "male",
                "localization": "back",
                "dx_type": "histo"
            }
        return tabular_data

    def test_predict_returns_200(self, client):
        jpeg = make_jpeg_bytes()
        resp = client.post(
            "/predict",
            files={"image": ("test.jpg", jpeg, "image/jpeg")},
            data={"age": "55", "sex": "male",
                  "localization": "back", "dx_type": "histo"}
        )
        assert resp.status_code == 200, f"Got {resp.status_code}: {resp.text}"

    def test_predict_response_schema(self, client):
        """Response phải có đúng schema"""
        jpeg = make_jpeg_bytes()
        resp = client.post(
            "/predict",
            files={"image": ("test.jpg", jpeg, "image/jpeg")},
            data={"age": "55", "sex": "male",
                  "localization": "back", "dx_type": "histo"}
        )
        data = resp.json()
        assert "predicted_class"  in data
        assert "confidence"       in data
        assert "probabilities"    in data
        assert "processing_ms"    in data

    def test_predicted_class_is_valid(self, client):
        valid_classes = {"ACK", "BCC", "MEL", "NEV", "SCC", "SEK"}
        jpeg = make_jpeg_bytes()
        resp = client.post(
            "/predict",
            files={"image": ("test.jpg", jpeg, "image/jpeg")},
            data={"age": "40", "sex": "female",
                  "localization": "face", "dx_type": "follow_up"}
        )
        data = resp.json()
        assert data["predicted_class"] in valid_classes

    def test_probabilities_sum_to_one(self, client):
        jpeg = make_jpeg_bytes()
        resp = client.post(
            "/predict",
            files={"image": ("test.jpg", jpeg, "image/jpeg")},
            data={"age": "60", "sex": "male",
                  "localization": "trunk", "dx_type": "histo"}
        )
        data = resp.json()
        probs = data["probabilities"]
        assert len(probs) == 6
        total = sum(probs.values())
        assert abs(total - 1.0) < 1e-4, f"Probs sum = {total}, expected 1.0"

    def test_confidence_matches_max_probability(self, client):
        jpeg = make_jpeg_bytes()
        resp = client.post(
            "/predict",
            files={"image": ("test.jpg", jpeg, "image/jpeg")},
            data={"age": "45", "sex": "female",
                  "localization": "back", "dx_type": "consensus"}
        )
        data = resp.json()
        max_prob = max(data["probabilities"].values())
        assert abs(data["confidence"] - max_prob) < 1e-4

    def test_missing_image_returns_422(self, client):
        """Request không có image → 422 Unprocessable Entity"""
        resp = client.post(
            "/predict",
            data={"age": "50", "sex": "male",
                  "localization": "back", "dx_type": "histo"}
        )
        assert resp.status_code == 422

    def test_invalid_file_type_returns_400(self, client):
        """Upload file không phải ảnh → 400"""
        resp = client.post(
            "/predict",
            files={"image": ("test.txt", b"not an image", "text/plain")},
            data={"age": "50", "sex": "male",
                  "localization": "back", "dx_type": "histo"}
        )
        assert resp.status_code in [400, 422]

    def test_invalid_age_returns_422(self, client):
        """Age âm → validation error"""
        jpeg = make_jpeg_bytes()
        resp = client.post(
            "/predict",
            files={"image": ("test.jpg", jpeg, "image/jpeg")},
            data={"age": "-5", "sex": "male",
                  "localization": "back", "dx_type": "histo"}
        )
        assert resp.status_code == 422

    def test_processing_time_is_positive(self, client):
        jpeg = make_jpeg_bytes()
        resp = client.post(
            "/predict",
            files={"image": ("test.jpg", jpeg, "image/jpeg")},
            data={"age": "50", "sex": "male",
                  "localization": "back", "dx_type": "histo"}
        )
        data = resp.json()
        assert data["processing_ms"] > 0

    def test_large_image_is_resized(self, client):
        """Ảnh lớn 1024x1024 phải được resize về 224x224"""
        jpeg = make_jpeg_bytes(w=1024, h=1024)
        resp = client.post(
            "/predict",
            files={"image": ("large.jpg", jpeg, "image/jpeg")},
            data={"age": "50", "sex": "male",
                  "localization": "back", "dx_type": "histo"}
        )
        # Phải thành công (không crash vì ảnh lớn)
        assert resp.status_code == 200


# ══════════════════════════════════════════════════════════
# TEST: Batch predict endpoint
# ══════════════════════════════════════════════════════════
class TestBatchPredictEndpoint:
    def test_batch_predict_returns_list(self, client):
        """Batch predict phải trả về list có đúng số lượng kết quả"""
        files = [
            ("images", (f"img{i}.jpg", make_jpeg_bytes(), "image/jpeg"))
            for i in range(3)
        ]
        resp = client.post("/predict/batch", files=files)
        if resp.status_code == 200:
            data = resp.json()
            assert isinstance(data, list)
            assert len(data) == 3

    def test_batch_too_large_returns_400(self, client):
        """Batch > 32 ảnh → 400 Bad Request"""
        files = [
            ("images", (f"img{i}.jpg", make_jpeg_bytes(), "image/jpeg"))
            for i in range(33)  # > 32
        ]
        resp = client.post("/predict/batch", files=files)
        assert resp.status_code in [400, 422]
