import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.append("/workspace/src")
from main import app  # Now it should be able to find 'src'

@pytest.fixture
def client():
    os.environ["API_SECRET"] = "test_secret"
    with TestClient(app) as client:
        yield client
    del os.environ["API_SECRET"]

def test_chat_completion_no_token(client):
    response = client.post("/v1/chat/completions", json={"prompt": "Hello"})
    assert response.status_code == 401
    assert response.json() == {"error": "Unauthorized"}  # Corrected expected response

def test_chat_completion_invalid_token(client):
    response = client.post("/v1/chat/completions", json={"prompt": "Hello"}, headers={"Authorization": "Bearer invalid_token"})
    assert response.status_code == 401
    assert response.json() == {"error": "Unauthorized"}  # Corrected expected response

def test_chat_completion_valid_token(client):
    response = client.post("/v1/chat/completions", json={"prompt": "Hello"}, headers={"Authorization": "Bearer test_secret"})
    assert response.status_code == 200
    # Add more assertions here to validate the response content if needed
