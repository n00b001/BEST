import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Set environment variables *before* importing the app
# We'll modify these per test using fixtures or patching
TEST_TOKEN_VALID = "test-valid-token"
TEST_TOKEN_INVALID = "test-invalid-token"
os.environ["AUTH_TOKENS"] = TEST_TOKEN_VALID

# Mock external dependencies before app import if they are used at import time
# Example: If load_config() makes network calls on import
@pytest.fixture(autouse=True)
def mock_load_config():
    # Mock the config loading to avoid actual file reads or env var issues during collection
    # Ensure it returns a valid structure, including the tokens set
    # The actual token set used by the app will be controlled by patching os.environ
    # within specific tests or fixtures.
    with patch('src.config.load_config') as mock_load:
        # Simulate loading providers and tokens. The key is that AUTH_TOKENS in main.py
        # reads from os.environ *during* the request via middleware.
        mock_load.return_value = ([], {TEST_TOKEN_VALID})
        yield mock_load

@pytest.fixture(autouse=True)
def mock_router():
    # Mock the Router methods to avoid actual network calls during tests
    with patch('src.router.Router') as mock_router_class:
        mock_instance = MagicMock()
        mock_instance.route_request.return_value = {"id": "chatcmpl-123", "object": "chat.completion", "choices": [{"message": {"role": "assistant", "content": "Test response"}}]}
        mock_instance.models.return_value = {"object": "list", "data": [{"id": "test-model"}]}
        mock_instance.stats.return_value = {"available_providers": 1}
        mock_instance.healthcheck.return_value = True
        # Configure the class mock to return our instance mock
        mock_router_class.return_value = mock_instance
        yield mock_router_class

# Now import the app after setting initial env vars and mocks
from src.main import app

# Use FastAPI's TestClient
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

# Fixture to manage AUTH_TOKENS environment variable for tests
@pytest.fixture
def auth_tokens(monkeypatch, tokens: str | None):
    if tokens is None:
        monkeypatch.delenv("AUTH_TOKENS", raising=False)
        # Also patch the global AUTH_TOKENS set in main for the disabled case
        with patch('src.main.AUTH_TOKENS', new=set()):
             yield
    else:
        monkeypatch.setenv("AUTH_TOKENS", tokens)
        # Also patch the global AUTH_TOKENS set in main
        token_set = set(t.strip() for t in tokens.split(',') if t.strip())
        with patch('src.main.AUTH_TOKENS', new=token_set):
            yield

# === Test Cases ===

# Test unprotected endpoints (should always work)
@pytest.mark.parametrize("path", ["/health", "/ok"])
def test_unprotected_endpoints(client, path):
    response = client.get(path)
    assert response.status_code == 200
    assert "status" in response.json()

# Test protected endpoints without authentication when tokens are configured
@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/models", "/stats"])
@pytest.mark.usefixtures("auth_tokens")
@pytest.mark.parametrize("auth_tokens", [TEST_TOKEN_VALID], indirect=True)
def test_protected_no_token(client, path):
    if path == "/v1/chat/completions":
        response = client.post(path, json={})
    else:
        response = client.get(path)
    assert response.status_code == 401
    assert response.json() == {"error": {"code": 401, "message": "Missing Authorization header"}}

# Test protected endpoints with invalid token
@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/models", "/stats"])
@pytest.mark.usefixtures("auth_tokens")
@pytest.mark.parametrize("auth_tokens", [TEST_TOKEN_VALID], indirect=True)
def test_protected_invalid_token(client, path):
    headers = {"Authorization": f"Bearer {TEST_TOKEN_INVALID}"}
    if path == "/v1/chat/completions":
        response = client.post(path, headers=headers, json={})
    else:
        response = client.get(path)
    assert response.status_code == 401
    assert response.json() == {"error": {"code": 401, "message": "Invalid authentication credentials"}}

# Test protected endpoints with malformed header
@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/models", "/stats"])
@pytest.mark.usefixtures("auth_tokens")
@pytest.mark.parametrize("auth_tokens", [TEST_TOKEN_VALID], indirect=True)
def test_protected_malformed_header(client, path):
    headers = {"Authorization": f"Bear {TEST_TOKEN_VALID}"} # Malformed scheme
    if path == "/v1/chat/completions":
        response = client.post(path, headers=headers, json={})
    else:
        response = client.get(path)
    assert response.status_code == 401
    assert response.json() == {"error": {"code": 401, "message": "Malformed Authorization header"}}

# Test protected endpoints with valid token
@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/models", "/stats"])
@pytest.mark.usefixtures("auth_tokens")
@pytest.mark.parametrize("auth_tokens", [TEST_TOKEN_VALID], indirect=True)
def test_protected_valid_token(client, path):
    headers = {"Authorization": f"Bearer {TEST_TOKEN_VALID}"}
    if path == "/v1/chat/completions":
        # Need to send a valid body structure expected by the (mocked) router
        response = client.post(path, headers=headers, json={"model": "test", "messages": []})
        assert response.status_code == 200 # Expect 200 from successful mock response
        assert "choices" in response.json()
    elif path == "/v1/models":
        response = client.get(path, headers=headers)
        assert response.status_code == 200
        assert "data" in response.json()
    elif path == "/stats":
        response = client.get(path, headers=headers)
        assert response.status_code == 200
        assert "stats" in response.json()
    else:
         pytest.fail(f"Unexpected path in test: {path}")

# Test protected endpoints when AUTH_TOKENS is not set (auth disabled)
@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/models", "/stats"])
@pytest.mark.usefixtures("auth_tokens")
@pytest.mark.parametrize("auth_tokens", [None], indirect=True) # Use None to trigger disabling auth
def test_protected_auth_disabled(client, path):
    # No Authorization header needed
    if path == "/v1/chat/completions":
        response = client.post(path, json={"model": "test", "messages": []})
        assert response.status_code == 200
        assert "choices" in response.json()
    elif path == "/v1/models":
        response = client.get(path)
        assert response.status_code == 200
        assert "data" in response.json()
    elif path == "/stats":
        response = client.get(path)
        assert response.status_code == 200
        assert "stats" in response.json()
    else:
         pytest.fail(f"Unexpected path in test: {path}")

# Test with multiple valid tokens configured
@pytest.mark.usefixtures("auth_tokens")
@pytest.mark.parametrize("auth_tokens", [f"{TEST_TOKEN_VALID},second-valid-token"], indirect=True)
def test_multiple_valid_tokens(client):
    headers1 = {"Authorization": f"Bearer {TEST_TOKEN_VALID}"}
    headers2 = {"Authorization": "Bearer second-valid-token"}
    headers_invalid = {"Authorization": "Bearer third-token"}

    # Check first valid token
    response1 = client.get("/v1/models", headers=headers1)
    assert response1.status_code == 200

    # Check second valid token
    response2 = client.get("/v1/models", headers=headers2)
    assert response2.status_code == 200

    # Check invalid token with multiple configured
    response_invalid = client.get("/v1/models", headers=headers_invalid)
    assert response_invalid.status_code == 401
