import pytest
import os
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi import status # Import status

from src.main import app # Import the main FastAPI app
from src.database import Base, get_db, APIKey, API_KEY_PREFIX
from src.config import load_config # To get current config if needed

# Use a separate in-memory SQLite database for admin API tests
DATABASE_URL_TEST_ADMIN = "sqlite:///./test_admin.db" # File based to persist across client calls if needed
engine_test_admin = create_engine(DATABASE_URL_TEST_ADMIN, connect_args={"check_same_thread": False})
SessionLocalTestAdmin = sessionmaker(autocommit=False, autoflush=False, bind=engine_test_admin)

# Override the get_db dependency for the app
def override_get_db():
    try:
        db = SessionLocalTestAdmin()
        yield db
    finally:
        db.close()

app.dependency_overrides[get_db] = override_get_db

# Set a test admin API key
TEST_ADMIN_API_KEY = "test_admin_secret_key"
os.environ["LLMGW_ADMIN_API_KEY"] = TEST_ADMIN_API_KEY 

# Fixture to manage database state for admin tests
@pytest.fixture(scope="module") # module scope as we use one DB for all admin tests
def admin_db_setup_module():
    Base.metadata.create_all(bind=engine_test_admin) # Create tables once per module
    yield
    Base.metadata.drop_all(bind=engine_test_admin) # Drop tables once after all tests in module

@pytest.fixture(scope="function") # function scope to clean data between tests
def admin_db_session(admin_db_setup_module): # Depends on module fixture
    db = SessionLocalTestAdmin()
    try:
        yield db # provide the session
    finally:
        # Clean up: delete all APIKey data after each test to ensure independence
        db.query(APIKey).delete()
        db.commit()
        db.close()

@pytest.fixture
def client(admin_db_session): # admin_db_session ensures DB is clean for each test
    # The app's get_db dependency is already overridden
    # Re-load config if necessary, though admin key is via env var for admin router
    # Ensure app.state.auth_config is set, if admin router relies on it (it shouldn't)
    # For main app auth config:
    # provider_configs, auth_cfg = load_config() # Load the actual auth config
    # app.state.auth_config = auth_cfg # Ensure it's on app.state for middleware
    # ^^^ This part might cause issues if config files are not where expected during test.
    # The admin router itself does not depend on app.state.auth_config from main app.
    # The APIKeyAuthMiddleware might, but admin routes should be exempt or have their own auth.
    # Let's ensure the main app.state.auth_config is at least initialized to avoid AttributeError
    if not hasattr(app.state, 'auth_config') or app.state.auth_config is None:
        _, auth_cfg = load_config() 
        app.state.auth_config = auth_cfg
    
    return TestClient(app)


def test_create_api_key_admin(client):
    key_name = "AdminCreatedKey"
    response = client.post(
        "/admin/keys",
        headers={"X-Admin-API-Key": TEST_ADMIN_API_KEY},
        json={"name": key_name}
    )
    assert response.status_code == status.HTTP_201_CREATED
    data = response.json()
    assert data["name"] == key_name
    assert "id" in data
    assert "plaintext_key" in data
    assert data["plaintext_key"].startswith(API_KEY_PREFIX)
    
    # Verify in DB (optional, but good for confidence)
    db = SessionLocalTestAdmin()
    key_in_db = db.query(APIKey).filter(APIKey.id == data["id"]).first()
    assert key_in_db is not None
    assert key_in_db.name == key_name
    db.close()

def test_create_api_key_admin_unauthorized(client):
    response = client.post(
        "/admin/keys",
        headers={"X-Admin-API-Key": "wrong_admin_key"},
        json={"name": "UnauthorizedKey"}
    )
    assert response.status_code == status.HTTP_403_FORBIDDEN

def test_list_api_keys_admin(client, admin_db_session): # use admin_db_session to interact with test DB
    # Create some keys directly in the test DB for listing
    from src.database import create_api_key as db_create_key
    key1_plain, _ = db_create_key(admin_db_session, name="ListKey1")
    key2_plain, _ = db_create_key(admin_db_session, name="ListKey2")

    response = client.get("/admin/keys", headers={"X-Admin-API-Key": TEST_ADMIN_API_KEY})
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2
    key_names = {item["name"] for item in data}
    assert "ListKey1" in key_names
    assert "ListKey2" in key_names
    for item in data:
        assert "hashed_key" not in item # Ensure sensitive data isn't exposed
        assert "plaintext_key" not in item

def test_list_api_keys_empty_admin(client):
    response = client.get("/admin/keys", headers={"X-Admin-API-Key": TEST_ADMIN_API_KEY})
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 0

def test_revoke_api_key_admin(client, admin_db_session):
    from src.database import create_api_key as db_create_key
    _, key_to_revoke = db_create_key(admin_db_session, name="RevokeThisKey")
    key_id_to_revoke = key_to_revoke.id

    response = client.delete(
        f"/admin/keys/{key_id_to_revoke}",
        headers={"X-Admin-API-Key": TEST_ADMIN_API_KEY}
    )
    assert response.status_code == status.HTTP_204_NO_CONTENT

    # Verify it's marked as revoked in DB
    admin_db_session.expire_all() # Expire all instances to ensure fresh data from DB
    revoked_key_in_db = admin_db_session.query(APIKey).filter(APIKey.id == key_id_to_revoke).first()
    assert revoked_key_in_db is not None
    assert revoked_key_in_db.revoked is True
    assert revoked_key_in_db.revoked_at is not None # Also check revoked_at

    # Try listing, it should not appear
    list_response = client.get("/admin/keys", headers={"X-Admin-API-Key": TEST_ADMIN_API_KEY})
    listed_keys = [item["id"] for item in list_response.json()]
    assert key_id_to_revoke not in listed_keys

def test_revoke_api_key_admin_not_found(client):
    response = client.delete(
        "/admin/keys/non-existent-id",
        headers={"X-Admin-API-Key": TEST_ADMIN_API_KEY}
    )
    assert response.status_code == status.HTTP_404_NOT_FOUND

def test_get_admin_usage_placeholder(client, admin_db_session):
    from src.database import create_api_key as db_create_key
    db_create_key(admin_db_session, name="UsageKey1")
    response = client.get("/admin/usage", headers={"X-Admin-API-Key": TEST_ADMIN_API_KEY})
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["name"] == "UsageKey1"
