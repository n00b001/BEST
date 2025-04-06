from src.config import ProviderConfig
from src.router import Router
from fastapi import HTTPException
import pytest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def router():
    providers = [
        ProviderConfig(
            base_url="http://example.com",
            api_key="test_key",
            model_name="test_model",
            priority={"overall_score": 1},
        )
    ]
    return Router(providers=providers)


def test_validate_token_success(router):
    os.environ["API_TOKEN"] = "test_token"
    request = {"headers": {"Authorization": "Bearer test_token"}}
    assert router._validate_token(request) == True


def test_validate_token_testing_success(router):
    request = {"headers": {"Authorization": "Bearer TESTING"}}
    assert router._validate_token(request) == True


def test_validate_token_missing(router):
    request = {"headers": {}}
    with pytest.raises(HTTPException) as excinfo:
        router._validate_token(request)
    assert excinfo.value.status_code == 401
    assert str(excinfo.value.detail) == "Authorization header missing"


def test_validate_token_invalid(router):
    os.environ["API_TOKEN"] = "test_token"
    request = {"headers": {"Authorization": "Bearer invalid_token"}}
    with pytest.raises(HTTPException) as excinfo:
        router._validate_token(request)
    assert excinfo.value.status_code == 401
    assert str(excinfo.value.detail) == "Invalid API token"
