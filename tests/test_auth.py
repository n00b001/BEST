import os
from fastapi.testclient import TestClient

from src.main import app

def test_auth():
    client = TestClient(app)
    os.environ["API_SECRET"] = "secret_token"

    # Test request without token
    response = client.post("/v1/chat/completions", json={"headers": {}}, headers={})
    assert response.status_code == 401
    assert response.json() == {"error": "Unauthorized"}

    # Test request with invalid token
    response = client.post("/v1/chat/completions", json={}, headers={"Authorization": "invalid_token"})
    assert response.status_code == 401
    assert response.json() == {"error": "Unauthorized"}
    headers={"Authorization": "invalid_token"}

    # Test request with valid token
    response = client.post("/v1/chat/completions", json={}, headers={"Authorization": "secret_token"})
    assert response.status_code != 401
