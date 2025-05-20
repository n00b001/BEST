import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.database import (
    Base,
    APIKey,
    create_api_key,
    get_api_key_by_id,
    get_api_key_by_raw_key,
    list_api_keys,
    revoke_api_key,
    update_api_key_last_used,
    verify_api_key,
    hash_api_key,
    API_KEY_PREFIX,
    generate_api_key_string
)

# Use an in-memory SQLite database for testing
DATABASE_URL_TEST = "sqlite:///:memory:"
engine_test = create_engine(DATABASE_URL_TEST, connect_args={"check_same_thread": False})
SessionLocalTest = sessionmaker(autocommit=False, autoflush=False, bind=engine_test)

@pytest.fixture(scope="function")
def db_session():
    # Create tables for each test function
    Base.metadata.create_all(bind=engine_test)
    db = SessionLocalTest()
    try:
        yield db
    finally:
        db.close()
        # Drop all tables after each test
        Base.metadata.drop_all(bind=engine_test)

def test_generate_api_key_string(db_session):
    full_key, raw_part = generate_api_key_string()
    assert full_key.startswith(API_KEY_PREFIX)
    assert len(raw_part) == 32 # UUID4 hex length
    assert full_key == API_KEY_PREFIX + raw_part

def test_hash_and_verify_api_key(db_session):
    _, raw_key = generate_api_key_string()
    hashed_key = hash_api_key(raw_key)
    assert hashed_key != raw_key
    assert verify_api_key(raw_key, hashed_key)
    assert not verify_api_key("wrong_key_part", hashed_key)

def test_create_and_get_api_key(db_session):
    key_name = "Test Key 1"
    plaintext_key, api_key_obj = create_api_key(db_session, name=key_name)

    assert plaintext_key is not None
    assert api_key_obj is not None
    assert api_key_obj.name == key_name
    assert api_key_obj.id is not None
    assert plaintext_key.startswith(API_KEY_PREFIX)
    
    raw_key_part = plaintext_key.replace(API_KEY_PREFIX, "")

    # Test get_api_key_by_raw_key
    retrieved_key_by_raw = get_api_key_by_raw_key(db_session, raw_key_part)
    assert retrieved_key_by_raw is not None
    assert retrieved_key_by_raw.id == api_key_obj.id
    assert retrieved_key_by_raw.name == key_name

    # Test get_api_key_by_id
    retrieved_key_by_id = get_api_key_by_id(db_session, api_key_obj.id)
    assert retrieved_key_by_id is not None
    assert retrieved_key_by_id.id == api_key_obj.id

def test_list_api_keys(db_session):
    keys_before = list_api_keys(db_session)
    assert len(keys_before) == 0

    create_api_key(db_session, name="Key A")
    create_api_key(db_session, name="Key B")

    keys_after = list_api_keys(db_session)
    assert len(keys_after) == 2
    assert set(k.name for k in keys_after) == {"Key A", "Key B"}

def test_revoke_api_key(db_session):
    _, api_key_obj = create_api_key(db_session, name="Revoke Test")
    assert api_key_obj.revoked is False

    # Revoke the key
    assert revoke_api_key(db_session, api_key_obj.id) is True
    
    retrieved_after_revoke = get_api_key_by_id(db_session, api_key_obj.id)
    assert retrieved_after_revoke is None # Should not be found by this function as it filters active keys

    # Verify raw key lookup also fails
    raw_key_part = _.replace(API_KEY_PREFIX, "") # _ still holds plaintext_key from create_api_key
    retrieved_raw_after_revoke = get_api_key_by_raw_key(db_session, raw_key_part)
    assert retrieved_raw_after_revoke is None

    # Check directly in DB
    revoked_key_direct = db_session.query(APIKey).filter(APIKey.id == api_key_obj.id).first()
    assert revoked_key_direct.revoked is True
    assert revoked_key_direct.revoked_at is not None

    # Try revoking again
    assert revoke_api_key(db_session, api_key_obj.id) is False # Already revoked

def test_update_api_key_last_used(db_session):
    _, api_key_obj = create_api_key(db_session, name="Last Used Test")
    assert api_key_obj.last_used is None

    assert update_api_key_last_used(db_session, api_key_obj.id) is True
    
    updated_key = db_session.query(APIKey).filter(APIKey.id == api_key_obj.id).first()
    assert updated_key.last_used is not None
    first_last_used = updated_key.last_used

    # Update again to see if timestamp changes
    import time; time.sleep(0.01) # Ensure time progresses enough for DB to register change
    assert update_api_key_last_used(db_session, api_key_obj.id) is True
    updated_key_again = db_session.query(APIKey).filter(APIKey.id == api_key_obj.id).first()
    assert updated_key_again.last_used > first_last_used

def test_get_non_existent_key(db_session):
    assert get_api_key_by_id(db_session, "non-existent-id") is None
    assert get_api_key_by_raw_key(db_session, "non_existent_raw_key") is None
