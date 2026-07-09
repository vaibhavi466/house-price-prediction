import pytest
from fastapi.testclient import TestClient

from server.main import app


@pytest.fixture(scope="module")
def client():
    """
    Fixtured test client. Utilizing the FastAPI TestClient within a lifespan context
    triggers the startup and shutdown handlers to load the model pipeline.
    """
    with TestClient(app) as c:
        yield c


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_get_locations(client):
    response = client.get("/locations")
    assert response.status_code == 200
    data = response.json()
    assert "locations" in data
    assert isinstance(data["locations"], list)
    # Whitefield is a standard frequent location in Bangalore and should be in the list
    assert "Whitefield" in data["locations"]


def test_predict_endpoint_success(client):
    payload = {"location": "Whitefield", "sqft": 1500.0, "bhk": 3, "bath": 3}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "predicted_price" in data
    assert isinstance(data["predicted_price"], float)
    assert data["predicted_price"] > 0.0

    assert "shap_contributions" in data
    assert isinstance(data["shap_contributions"], list)
    assert len(data["shap_contributions"]) <= 3
    for contrib in data["shap_contributions"]:
        assert "feature" in contrib
        assert "contribution" in contrib
        assert isinstance(contrib["feature"], str)
        assert isinstance(contrib["contribution"], float)


def test_predict_endpoint_unseen_location(client):
    payload = {
        "location": "Antigravity Research Base",  # Unseen location
        "sqft": 1200.0,
        "bhk": 2,
        "bath": 2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_price"] > 0.0


def test_predict_validation_errors(client):
    # Case 1: Negative sqft
    payload = {"location": "Whitefield", "sqft": -500.0, "bhk": 2, "bath": 2}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

    # Case 2: Extreme BHK (>20)
    payload = {"location": "Whitefield", "sqft": 1500.0, "bhk": 25, "bath": 2}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

    # Case 3: Empty location
    payload = {"location": "   ", "sqft": 1500.0, "bhk": 2, "bath": 2}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
