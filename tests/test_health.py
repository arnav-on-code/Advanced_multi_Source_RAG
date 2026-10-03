from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


# =========================================================
# LIVENESS
# =========================================================


def test_health_check():
    """Liveness endpoint should confirm that the API is running."""

    response = client.get("/health")

    assert response.status_code == 200

    assert response.headers["content-type"].startswith(
        "application/json"
    )

    data = response.json()

    assert data["status"] == "healthy"
    assert data["service"]
    assert data["version"]
    assert data["environment"]


# =========================================================
# READINESS
# =========================================================


def test_readiness_check():
    """Readiness endpoint should return a valid readiness response."""

    response = client.get("/health/ready")

    assert response.status_code == 200

    assert response.headers["content-type"].startswith(
        "application/json"
    )

    data = response.json()

    assert data["status"] == "ready"
    assert data["service"]
    assert data["version"]
    assert data["environment"]


# =========================================================
# RESPONSE CONTRACT
# =========================================================


def test_health_response_structure():
    """Health response should contain the expected API contract."""

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert set(data.keys()) >= {
        "status",
        "service",
        "version",
        "environment",
    }


def test_readiness_response_structure():
    """Readiness response should contain the expected API contract."""

    response = client.get("/health/ready")

    assert response.status_code == 200

    data = response.json()

    assert set(data.keys()) >= {
        "status",
        "service",
        "version",
        "environment",
    }


# =========================================================
# JSON RESPONSE
# =========================================================


def test_health_response_is_json():
    """Health endpoint should return JSON."""

    response = client.get("/health")

    assert response.status_code == 200

    assert response.headers["content-type"].startswith(
        "application/json"
    )

    assert isinstance(
        response.json(),
        dict,
    )


def test_readiness_response_is_json():
    """Readiness endpoint should return JSON."""

    response = client.get("/health/ready")

    assert response.status_code == 200

    assert response.headers["content-type"].startswith(
        "application/json"
    )

    assert isinstance(
        response.json(),
        dict,
    )