import logging
import os
from contextlib import asynccontextmanager

import coloredlogs
import requests
import uvicorn
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, HTTPException, Request, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response


# Assuming AUTH_TOKENS is loaded and available from config
from src.config import load_config, AUTH_TOKENS
from src.consts import PORT, LOG_LEVEL, EXTERNAL_HEALTHCHECK_URL
from src.router import Router
from src.utils import truncate_dict

logger = logging.getLogger(__name__)
coloredlogs.install(level=LOG_LEVEL, logger=logger)

# --- External Health Check ---
def external_health_check():
    try:
        response = requests.get(EXTERNAL_HEALTHCHECK_URL, timeout=5) # Added timeout
        response.raise_for_status()
        if not response.ok:
            raise RuntimeError(f"status_code: {response.status_code} text {response.text}")
        logger.debug(f"External health check to {EXTERNAL_HEALTHCHECK_URL} successful.")
    except requests.exceptions.RequestException as e:
        logger.error(f"External health check failed for {EXTERNAL_HEALTHCHECK_URL}: {e}")
    except Exception as e:
         logger.error(f"An unexpected error occurred during external health check: {e}")


# --- Application Lifespan (Context Manager) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load config and router
    providers, loaded_tokens = load_config() # Reload in case .env changed
    # Update the global AUTH_TOKENS set (this is crucial for the middleware)
    global AUTH_TOKENS
    AUTH_TOKENS = loaded_tokens
    app.state.router = Router(providers) # Pass only providers to Router
    logger.info(f"Router initialized with {len(providers)} providers.")
    if not AUTH_TOKENS:
        logger.warning("Authentication is disabled as no AUTH_TOKENS are configured.")
    else:
        logger.info(f"Authentication enabled. {len(AUTH_TOKENS)} valid token(s) loaded.")


    # Setup background task for external health check
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        external_health_check,
        "interval",
        minutes=10,
        id="external_health_check_job", # Added job ID
        replace_existing=True # Avoid duplicate jobs on potential reloads
    )
    scheduler.start()
    logger.info("Background scheduler started for external health checks.")

    yield

    # --- Cleanup ---
    scheduler.shutdown()
    logger.info("Background scheduler shut down.")


# --- FastAPI App Initialization ---
app = FastAPI(lifespan=lifespan)

# --- CORS Middleware ---
# Needs to be before AuthMiddleware if cross-origin requests need authentication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Consider restricting this in production
    allow_credentials=True, # Important for Authorization header
    allow_methods=["*"],
    allow_headers=["*"], # Ensure 'Authorization' is allowed if origins are restricted
)

# --- Authentication Middleware ---
class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Skip auth for health/ok endpoints
        if request.url.path in ["/health", "/ok"]:
            return await call_next(request)

        # If no tokens are configured, bypass authentication (log warning on startup)
        if not AUTH_TOKENS:
             return await call_next(request)

        # Try to get the authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            logger.warning(f"Unauthorized: Missing Authorization header. Path: {request.url.path}")
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"error": {"code": 401, "message": "Missing Authorization header"}},
                headers={"WWW-Authenticate": "Bearer"}, # Standard practice
            )

        # Attempt to validate the Bearer token
        try:
            scheme, credentials = auth_header.split()
            if scheme.lower() != 'bearer':
                 raise ValueError("Invalid scheme")
            if credentials not in AUTH_TOKENS:
                logger.warning(f"Unauthorized: Invalid token provided. Path: {request.url.path}")
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"error": {"code": 401, "message": "Invalid authentication credentials"}},
                    headers={"WWW-Authenticate": "Bearer realm='Restricted Area'"}, # Standard practice
                )
            # Token is valid, proceed with the request
            # logger.debug(f"Authorized access. Path: {request.url.path}") # Optional: log successful auth
            return await call_next(request)

        except ValueError:
            logger.warning(f"Unauthorized: Malformed Authorization header. Path: {request.url.path}")
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"error": {"code": 401, "message": "Malformed Authorization header"}},
                 headers={"WWW-Authenticate": "Bearer"}, # Standard practice
            )
        except Exception as e: # Catch unexpected errors during auth
            logger.error(f"Unexpected error during authentication for path {request.url.path}: {e}", exc_info=True)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"error": {"code": 500, "message": "Internal server error during authentication"}}
            )

# Add the authentication middleware *after* CORS
app.add_middleware(AuthMiddleware)


# --- API Endpoints ---
# No Depends(verify_token) needed anymore.

@app.post("/chat/completions")
@app.post("/v1/chat/completions")
async def chat_completion(request: Request): # Changed from dict to Request to access state if needed
    router: Router = request.app.state.router
    try:
        request_body = await request.json() # Get JSON body
        trunc_request = await truncate_dict(request_body)
        logger.info(f"Got request: {trunc_request}")
        content = await router.route_request(request_body)
        return content
    except HTTPException as e:
        # Log the exception detail captured by the router or middleware
        logger.error(f"HTTPException in chat_completion: {e.detail}", exc_info=True)
        # Ensure response is JSON with the error detail
        return JSONResponse(status_code=e.status_code, content={"error": {"message": e.detail, "code": e.status_code}})
    except Exception as e: # Catch unexpected errors in this endpoint handler
        logger.error(f"Unexpected error in chat_completion: {e}", exc_info=True)
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"error": {"message": "Internal Server Error", "code": 500}})


@app.get("/health")
async def health_check(request: Request): # Changed to Request
    router: Router = request.app.state.router
    try:
        content = await router.healthcheck()
        return {"status": content}
    except HTTPException as e:
        logger.error(f"HTTPException in health_check: {e.detail}", exc_info=True)
        return JSONResponse(status_code=e.status_code, content={"error": {"message": e.detail, "code": e.status_code}})
    except Exception as e: # Catch unexpected errors
        logger.error(f"Unexpected error in health_check: {e}", exc_info=True)
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"error": {"message": "Internal Server Error", "code": 500}})


@app.get("/ok")
async def ok():
    return {"status": "ok"}


@app.get("/stats")
async def stats(request: Request): # Changed to Request
    router: Router = request.app.state.router
    try:
        content = await router.stats()
        return {"stats": content}
    except HTTPException as e:
        logger.error(f"HTTPException in stats: {e.detail}", exc_info=True)
        return JSONResponse(status_code=e.status_code, content={"error": {"message": e.detail, "code": e.status_code}})
    except Exception as e: # Catch unexpected errors
        logger.error(f"Unexpected error in stats: {e}", exc_info=True)
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"error": {"message": "Internal Server Error", "code": 500}})


@app.get("/models")
@app.get("/v1/models")
async def models(request: Request): # Changed to Request
    router: Router = request.app.state.router
    try:
        content = await router.models()
        return content
    except HTTPException as e:
        logger.error(f"HTTPException in models: {e.detail}", exc_info=True)
        return JSONResponse(status_code=e.status_code, content={"error": {"message": e.detail, "code": e.status_code}})
    except Exception as e: # Catch unexpected errors
        logger.error(f"Unexpected error in models: {e}", exc_info=True)
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"error": {"message": "Internal Server Error", "code": 500}})


# --- Main Execution ---
if __name__ == "__main__":
    # Ensure config (and tokens) are loaded before uvicorn starts workers
    # This initial load helps catch config issues early, lifespan handles the rest
    load_config()
    logger.info(f"Starting server on port {PORT}")
    # Use reload=True only if DEBUG_MODE is explicitly set to 'true' or '1'
    debug_mode_enabled = os.getenv("DEBUG_MODE", "false").lower() in ["true", "1", "t"]
    uvicorn.run("src.main:app", host="0.0.0.0", port=PORT, reload=debug_mode_enabled)

