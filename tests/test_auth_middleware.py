import pytest
from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware import Middleware
from starlette.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


from src.auth_middleware import APIKeyAuthMiddleware, EXEMPT_PATHS
from src.config import AuthConfig
from src.database import create_api_key, SessionLocal, Base, API_KEY_PREFIX, get_db as get_actual_db, revoke_api_key

# Use a separate in-memory SQLite database for these tests too
DATABASE_URL_TEST_MW = "sqlite:///./test_mw.db" # Use file based to avoid issues with TestClient context
engine_test_mw = create_engine(DATABASE_URL_TEST_MW, connect_args={"check_same_thread": False})
SessionLocalTestMw = sessionmaker(autocommit=False, autoflush=False, bind=engine_test_mw)

# Mock app state for tests
class MockAppState:
    def __init__(self, auth_config: AuthConfig):
        self.auth_config = auth_config

# Test endpoint
async def test_endpoint(request: Request):
    if hasattr(request.state, "api_key_id"): # Check for api_key_id
        return JSONResponse({
            "message": "success", 
            "key_id": request.state.api_key_id,
            "key_name": request.state.api_key_name # Also retrieve name
        })
    return JSONResponse({"message": "success, no key"})

@pytest.fixture(scope="module")
def test_app_factory():
    # Factory to create app instances with different middleware configs
    def _create_app(auth_enabled: bool, initial_keys: list = None):
        Base.metadata.create_all(bind=engine_test_mw) # Create tables in the test DB
        
        # Populate initial keys if any
        if initial_keys:
            db = SessionLocalTestMw()
            for key_name in initial_keys:
                create_api_key(db, name=key_name)
            db.close()

        app = FastAPI(
            middleware=[Middleware(APIKeyAuthMiddleware)],
        )
        app.state = MockAppState(AuthConfig(authentication_enabled=auth_enabled))
        
        app.add_route("/test-protected", test_endpoint)
        for path in EXEMPT_PATHS:
            # Ensure paths are actual routes for testing exemption
            if path == "/admin": # prefix
                 app.add_route("/admin/somepath", lambda r: JSONResponse({"message": "admin exempt"}), methods=["GET"])
            # These are auto-handled by FastAPI or not simple GET routes
            elif path not in ["/docs", "/openapi.json"]: 
                 app.add_route(path, lambda r: JSONResponse({"message": "exempt"}), methods=["GET"])
        return app

    yield _create_app

    # Teardown: drop all tables from the test_mw.db
    Base.metadata.drop_all(bind=engine_test_mw)


@pytest.fixture(scope="function")
def db_session_mw():
    # Provides a session for setting up keys IN THE TEST_MW database
    Base.metadata.create_all(bind=engine_test_mw)
    db = SessionLocalTestMw()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine_test_mw) # Clean after each test for isolation

def test_auth_disabled(test_app_factory):
    app = test_app_factory(auth_enabled=False)
    client = TestClient(app)
    response = client.get("/test-protected")
    assert response.status_code == status.HTTP_200_OK
    assert response.json()["message"] == "success, no key"

def test_auth_enabled_no_key_provided(test_app_factory):
    app = test_app_factory(auth_enabled=True)
    client = TestClient(app)
    response = client.get("/test-protected")
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert response.json()["detail"] == "Not authenticated"

def test_auth_enabled_invalid_header_format(test_app_factory):
    app = test_app_factory(auth_enabled=True)
    client = TestClient(app)
    response = client.get("/test-protected", headers={"Authorization": "InvalidKey"})
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert response.json()["detail"] == "Invalid authentication credentials"

def test_auth_enabled_key_bad_prefix(test_app_factory, db_session_mw):
    app = test_app_factory(auth_enabled=True)
    client = TestClient(app)
    response = client.get("/test-protected", headers={"Authorization": "Bearer badprefix-123"})
    assert response.status_code == status.HTTP_403_FORBIDDEN
    assert response.json()["detail"] == "Invalid API key format"
    
def test_auth_enabled_key_not_found(test_app_factory, db_session_mw):
    app = test_app_factory(auth_enabled=True) 
    client = TestClient(app)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("src.auth_middleware.SessionLocal", SessionLocalTestMw)
        response = client.get("/test-protected", headers={"Authorization": f"Bearer {API_KEY_PREFIX}nonexistentkey"})
        assert response.status_code == status.HTTP_403_FORBIDDEN 
        assert response.json()["detail"] == "Invalid or revoked API key"


def test_auth_enabled_valid_key(test_app_factory, db_session_mw):
    key_name = "middleware_test_key"
    plaintext_key, key_obj = create_api_key(db_session_mw, name=key_name) # Get key_obj
    
    app = test_app_factory(auth_enabled=True) 
    client = TestClient(app)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("src.auth_middleware.SessionLocal", SessionLocalTestMw)
        response = client.get("/test-protected", headers={"Authorization": f"Bearer {plaintext_key}"})
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["message"] == "success"
        assert data["key_id"] == key_obj.id # Assert against key_obj.id
        assert data["key_name"] == key_name # Assert against key_name

def test_auth_enabled_revoked_key(test_app_factory, db_session_mw):
    key_name = "revoked_key_for_mw"
    plaintext_key, key_obj = create_api_key(db_session_mw, name=key_name)
    revoke_api_key(db_session_mw, key_obj.id)

    app = test_app_factory(auth_enabled=True)
    client = TestClient(app)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("src.auth_middleware.SessionLocal", SessionLocalTestMw)
        response = client.get("/test-protected", headers={"Authorization": f"Bearer {plaintext_key}"})
        assert response.status_code == status.HTTP_403_FORBIDDEN
        assert response.json()["detail"] == "Invalid or revoked API key"

@pytest.mark.parametrize("path", ["/health", "/ok", "/admin/somepath"]) 
def test_exempt_paths_auth_enabled(test_app_factory, path):
    app = test_app_factory(auth_enabled=True)
    client = TestClient(app)
    response = client.get(path)
    assert response.status_code == status.HTTP_200_OK
    if path == "/admin/somepath":
        assert response.json()["message"] == "admin exempt"
    else:
        assert response.json()["message"] == "exempt"
