import os
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader
from sqlalchemy.orm import Session

from src.database import (
    APIKey,
    create_api_key,
    get_db,
    list_api_keys,
    revoke_api_key,
    get_api_key_by_id,
    API_KEY_PREFIX
)
from pydantic import BaseModel, Field # For request/response models

# --- Configuration for Admin Auth ---
# For simplicity, using a fixed API key read from an environment variable.
# In a production system, you might want a more robust admin auth mechanism.
# ADMIN_API_KEY = os.getenv("LLMGW_ADMIN_API_KEY", "default_admin_key_please_change") # Moved
API_KEY_NAME = "X-Admin-API-Key"
admin_api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

router = APIRouter(prefix="/admin", tags=["Admin"])

async def get_admin_api_key(api_key_header: str = Security(admin_api_key_header)):
    # Fetch the admin API key from environment variable at runtime
    admin_api_key_from_env = os.getenv("LLMGW_ADMIN_API_KEY", "default_admin_key_please_change")
    if api_key_header == admin_api_key_from_env:
        return api_key_header
    else:
        # Log the mismatched keys for easier debugging if this happens in prod/staging
        # logger.warning(f"Admin auth failed. Provided: '{api_key_header}', Expected: '{admin_api_key_from_env}'")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate admin credentials",
        )

# --- Pydantic Models for Admin API ---
import datetime # Import datetime

class APIKeyCreateRequest(BaseModel):
    name: str = Field(..., description="A user-friendly name for the API key.")

class APIKeyCreatedResponse(BaseModel):
    id: str
    name: str
    created_at: datetime.datetime # Changed to datetime
    plaintext_key: str = Field(..., description="The generated API key. Store this securely, it won't be shown again.")

class APIKeyDetails(BaseModel):
    id: str
    name: str
    created_at: datetime.datetime # Changed to datetime
    last_used: Optional[datetime.datetime] = None # Changed to datetime
    revoked: bool
    revoked_at: Optional[datetime.datetime] = None # Changed to datetime
    # Do NOT include hashed_key in responses

    model_config = {'from_attributes': True} # For Pydantic v2

@router.post("/keys", response_model=APIKeyCreatedResponse, status_code=status.HTTP_201_CREATED)
async def handle_create_api_key(
    payload: APIKeyCreateRequest,
    db: Session = Depends(get_db),
    admin_key: str = Depends(get_admin_api_key) # Protects the endpoint
):
    """Generate a new API key."""
    plaintext_key, db_api_key = create_api_key(db, name=payload.name)
    if not db_api_key or not plaintext_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create API key."
        )
    return APIKeyCreatedResponse(
        id=db_api_key.id,
        name=db_api_key.name,
        created_at=db_api_key.created_at, # Pass datetime object directly
        plaintext_key=plaintext_key
    )

@router.get("/keys", response_model=List[APIKeyDetails])
async def handle_list_api_keys(
    db: Session = Depends(get_db),
    admin_key: str = Depends(get_admin_api_key)
):
    """List all active API keys."""
    keys = list_api_keys(db)
    # Convert to Pydantic model, ensuring datetimes are nicely formatted
    return [APIKeyDetails.from_orm(key) for key in keys]


@router.delete("/keys/{key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def handle_revoke_api_key(
    key_id: str,
    db: Session = Depends(get_db),
    admin_key: str = Depends(get_admin_api_key)
):
    """Revoke a specific API key by its ID."""
    key_obj = get_api_key_by_id(db, key_id) # Check if key exists and is not revoked
    if not key_obj:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="API Key not found or already revoked.")

    if not revoke_api_key(db, key_id):
        # This might happen if the key was deleted/revoked between the get and revoke call,
        # or if there's a DB error.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to revoke API key."
        )
    return None # FastAPI handles 204 No Content response

@router.get("/usage", summary="View usage statistics by key (Placeholder)")
async def handle_get_usage_stats(
    db: Session = Depends(get_db),
    admin_key: str = Depends(get_admin_api_key)
):
    """
    Placeholder for viewing usage statistics by key.
    Currently returns basic key information including last_used.
    """
    keys = list_api_keys(db) # Or query all keys including revoked if needed for historical data
    return [APIKeyDetails.from_orm(key) for key in keys]
