import logging
from typing import Callable, Awaitable # Import Callable and Awaitable
from fastapi import Request, status, Response # Import Response for type hinting
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp, Receive, Scope, Send # Keep existing starlette.types imports

from src.database import SessionLocal, get_api_key_by_raw_key, update_api_key_last_used, API_KEY_PREFIX
from src.config import AuthConfig # Assuming AuthConfig is accessible

logger = logging.getLogger(__name__)

# Define paths that should skip authentication
# Admin paths will eventually need their own protection, but not via this user API key middleware.
EXEMPT_PATHS = [
    "/docs",  # OpenAPI UI
    "/openapi.json", # OpenAPI schema
    "/health",
    "/ok",
    "/admin", # Prefix for all admin routes
]

class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp): # Remove auth_config from __init__
        super().__init__(app)
        # self.auth_config is no longer set here

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]): # Updated type hint
        auth_config: AuthConfig = request.app.state.auth_config # Get auth_config from app state
            
        # Check if the current path is exempt or if auth is disabled
        if not auth_config.authentication_enabled or \
           any(request.url.path.startswith(exempt_path) for exempt_path in EXEMPT_PATHS):
            response = await call_next(request)
            return response

        authorization_header = request.headers.get("Authorization")
        if not authorization_header:
            logger.warning("Missing Authorization header")
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Not authenticated"},
            )

        parts = authorization_header.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            logger.warning(f"Invalid Authorization header format: {authorization_header}")
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Invalid authentication credentials"},
            )

        token = parts[1]
        if not token.startswith(API_KEY_PREFIX):
            logger.warning(f"Invalid API key prefix for token: {token[:len(API_KEY_PREFIX)+5]}...")
            return JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={"detail": "Invalid API key format"},
            )

        raw_key_part = token[len(API_KEY_PREFIX):]

        db = SessionLocal()
        try:
            api_key_obj = get_api_key_by_raw_key(db, raw_key_part)
            if not api_key_obj:
                logger.warning(f"Invalid or revoked API key used: {token[:len(API_KEY_PREFIX)+5]}...")
                return JSONResponse(
                    status_code=status.HTTP_403_FORBIDDEN,
                    content={"detail": "Invalid or revoked API key"},
                )

            # Store key details (not the ORM object directly) in request state
            # to avoid detached instance issues if accessed after session closes.
            request.state.api_key_id = api_key_obj.id
            request.state.api_key_name = api_key_obj.name # Store name if needed by endpoints
            # For this test, we only need id, but storing name is a good practice if other endpoints use it.
            
            # Update last used timestamp
            # Consider if this should be done before or after the request processing,
            # and if it's too much I/O for every request. For now, updating it here.
            update_api_key_last_used(db, api_key_obj.id)

        except Exception as e:
            # Log the exception e
            logger.error(f"Error during API key validation: {e}")
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Internal server error during authentication"},
            )
        finally:
            db.close()

        response = await call_next(request)
        return response
