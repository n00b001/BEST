import uuid
import datetime
from typing import Optional, Tuple, List

from sqlalchemy import create_engine, Column, String, DateTime, Boolean, inspect
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError
from passlib.context import CryptContext

# --- Configuration ---
DATABASE_URL = "sqlite:///./llm_gateway.db"
# In a real application, consider making the secret key configurable and securely managed.
SECRET_KEY = "your-super-secret-key" # Used for deriving sub-keys for hashing if needed, or as part of salt.
API_KEY_PREFIX = "llmgw-sk-"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- Password Hashing ---
# Using passlib for secure password hashing
# bcrypt is a good default choice.
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- Models ---
class APIKey(Base):
    __tablename__ = "api_keys"

    id = Column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    hashed_key = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False) # User-friendly name for the key
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    last_used = Column(DateTime, nullable=True)
    revoked = Column(Boolean, default=False)
    revoked_at = Column(DateTime, nullable=True)

    def __repr__(self):
        return f"<APIKey(id={self.id}, name='{self.name}', revoked={self.revoked})>"

# --- Database Initialization ---
def create_db_and_tables():
    # Check if tables exist before creating
    inspector = inspect(engine)
    if not inspector.has_table(APIKey.__tablename__):
        Base.metadata.create_all(bind=engine)
        print("Database tables created.")
    else:
        print("Database tables already exist.")

# --- CRUD Operations for API Keys ---

def generate_api_key_string() -> Tuple[str, str]:
    """Generates a new random API key string and its part to be hashed."""
    # Generate a secure random key part
    raw_key = uuid.uuid4().hex
    # Prepend the prefix to the raw key for user display
    full_key_display = f"{API_KEY_PREFIX}{raw_key}"
    return full_key_display, raw_key

def hash_api_key(key_to_hash: str) -> str:
    """Hashes the raw part of the API key using passlib."""
    return pwd_context.hash(key_to_hash)

def verify_api_key(plain_key_to_check: str, hashed_key_from_db: str) -> bool:
    """Verifies a plain API key against a hashed key from the database."""
    return pwd_context.verify(plain_key_to_check, hashed_key_from_db)

def create_api_key(db_session, name: str) -> Tuple[Optional[str], Optional[APIKey]]:
    """
    Generates a new API key, hashes its raw part, stores it in the database,
    and returns the full plaintext key (including prefix) and the APIKey ORM object.
    Returns (None, None) on failure.
    """
    try:
        full_key_display, raw_key_part = generate_api_key_string()
        hashed = hash_api_key(raw_key_part)

        db_api_key = APIKey(
            hashed_key=hashed,
            name=name
        )
        db_session.add(db_api_key)
        db_session.commit()
        db_session.refresh(db_api_key)
        return full_key_display, db_api_key
    except SQLAlchemyError as e:
        db_session.rollback()
        # In a real app, you'd log this error.
        print(f"Error creating API key: {e}")
        return None, None

def get_api_key_by_raw_key(db_session, raw_key_to_lookup: str) -> Optional[APIKey]:
    """
    Retrieves a non-revoked API key by its raw (unprefixed, unhashed) part.
    This involves iterating and verifying, as we only store hashed keys.
    This function should be used carefully due to performance implications on large datasets.
    A more direct lookup (e.g., by a prefix of the hash or a separate lookup token) might be
    preferable in high-load scenarios if direct key lookup is frequent.
    However, for validating an incoming key, this is the correct approach.
    """
    # This function is a bit tricky. We can't directly query by the raw_key_to_lookup
    # because we only store hashes. We must fetch potential keys and verify.
    # For simplicity and security, we iterate. If performance becomes an issue,
    # consider adding a non-sensitive, indexed lookup hint if possible,
    # or re-evaluate how keys are identified before hashing.

    # This simplified version assumes `get_api_key_by_hashed_key` or similar would be more typical for direct lookup.
    # The common use case is: user provides `llmgw-sk-abc123`, middleware extracts `abc123`,
    # then this function is called with `abc123`.
    
    # A truly secure direct lookup by the key value is not possible if keys are properly hashed with salt.
    # We must iterate through keys and use verify_api_key.
    # This is generally acceptable for checking an incoming API key from a user.

    keys = db_session.query(APIKey).filter(APIKey.revoked == False).all()
    for key_obj in keys:
        if verify_api_key(raw_key_to_lookup, key_obj.hashed_key):
            return key_obj
    return None


def get_api_key_by_id(db_session, key_id: str) -> Optional[APIKey]:
    """Retrieves an API key by its ID."""
    try:
        return db_session.query(APIKey).filter(APIKey.id == key_id, APIKey.revoked == False).first()
    except SQLAlchemyError: # Add specific exceptions if known
        # Log error
        return None

def list_api_keys(db_session) -> List[APIKey]:
    """Lists all non-revoked API keys."""
    try:
        return db_session.query(APIKey).filter(APIKey.revoked == False).all()
    except SQLAlchemyError:
        # Log error
        return []

def revoke_api_key(db_session, key_id: str) -> bool:
    """Marks an API key as revoked. Returns True on success, False on failure."""
    try:
        db_key = db_session.query(APIKey).filter(APIKey.id == key_id).first()
        if db_key and not db_key.revoked:
            db_key.revoked = True
            db_key.revoked_at = datetime.datetime.utcnow()
            db_session.commit()
            return True
        return False # Key not found or already revoked
    except SQLAlchemyError:
        db_session.rollback()
        # Log error
        return False

def update_api_key_last_used(db_session, key_id: str) -> bool:
    """Updates the last_used timestamp for an API key. Returns True on success."""
    try:
        db_key = db_session.query(APIKey).filter(APIKey.id == key_id).first()
        if db_key:
            db_key.last_used = datetime.datetime.utcnow()
            db_session.commit()
            return True
        return False
    except SQLAlchemyError:
        db_session.rollback()
        # Log error
        return False

# --- Helper to get DB session (dependency for FastAPI) ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

if __name__ == "__main__":
    # This is for basic testing and table creation when running the script directly.
    print("Initializing database and creating tables if they don't exist...")
    create_db_and_tables()
    print("Database setup complete.")

    # Example Usage (Optional: for testing directly)
    # To run this, you'd typically do it in a context where you can manage sessions.
    # For example, within a FastAPI route or a dedicated script.
    # Session = sessionmaker(bind=engine)
    # session = Session()
    #
    # # Create a key
    # new_key_display, new_key_obj = create_api_key(session, name="Test Key 1")
    # if new_key_display and new_key_obj:
    #     print(f"Created key: {new_key_display}, ID: {new_key_obj.id}")
    #
    #     # Extract raw part for verification
    #     raw_part_to_verify = new_key_display.replace(API_KEY_PREFIX, "")
    #
    #     # Verify
    #     retrieved_key = get_api_key_by_raw_key(session, raw_part_to_verify)
    #     if retrieved_key:
    #         print(f"Verified and retrieved key: {retrieved_key.name}")
    #         assert retrieved_key.id == new_key_obj.id
    #
    #         # List keys
    #         all_keys = list_api_keys(session)
    #         print(f"All keys: {[k.name for k in all_keys]}")
    #
    #         # Revoke key
    #         revoke_api_key(session, new_key_obj.id)
    #         print(f"Revoked key: {new_key_obj.id}")
    #
    #         retrieved_after_revoke = get_api_key_by_id(session, new_key_obj.id)
    #         assert retrieved_after_revoke is None # Should not be found by this function
    #
    #         # Check if it's truly marked as revoked if fetched directly without active check
    #         revoked_key_direct = session.query(APIKey).filter(APIKey.id == new_key_obj.id).first()
    #         print(f"Revoked status from DB: {revoked_key_direct.revoked}")
    #         assert revoked_key_direct.revoked is True
    #
    # else:
    #     print("Failed to create key.")
    #
    # session.close()
