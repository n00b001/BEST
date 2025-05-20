import logging
from contextlib import asynccontextmanager

import coloredlogs
import requests
import uvicorn
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, HTTPException, Request, status # Added Request and status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import load_config, AuthConfig # AuthConfig is explicitly imported for type hinting
from src.consts import PORT, LOG_LEVEL, EXTERNAL_HEALTHCHECK_URL
from src.router import Router
from src.utils import truncate_dict
from src.database import create_db_and_tables # Added import
from src.auth_middleware import APIKeyAuthMiddleware # Added import
from src.admin_router import router as admin_router # <<< ADD THIS

logger = logging.getLogger(__name__)
coloredlogs.install(level=LOG_LEVEL, logger=logger)


def external_health_check():
    response = requests.get(EXTERNAL_HEALTHCHECK_URL)
    response.raise_for_status()
    if not response.ok:
        raise RuntimeError(f"status_code: {response.status_code} text {response.text}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        external_health_check,
        "interval",
        minutes=10,
    )
    scheduler.start()

    logger.info("Initializing database...") # Added logging
    create_db_and_tables() # Ensured this call is present
    logger.info("Database initialization complete.") # Added logging

    # app.state.router = Router(load_config()) # OLD
    provider_configs, auth_config = load_config()
    app.state.router = Router(provider_configs) # For now, Router only takes provider_configs
    app.state.auth_config = auth_config # Store auth_config in app.state
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add the APIKeyAuthMiddleware
app.add_middleware(APIKeyAuthMiddleware)

app.include_router(admin_router) # <<< ADD THIS to mount the admin routes


@app.post("/chat/completions")
@app.post("/v1/chat/completions")
async def chat_completion(request: Request): # <--- Change to accept the whole Request object
    router: Router = request.app.state.router # Corrected: app.state -> request.app.state
    auth_config: AuthConfig = request.app.state.auth_config # Corrected: app.state -> request.app.state

    # Extract the actual request body for the router
    request_body = await request.json()

    try:
        # Log API key usage if auth is enabled
        if auth_config.authentication_enabled:
            if hasattr(request.state, "api_key") and request.state.api_key:
                api_key_id = request.state.api_key.id
                logger.info(f"Request to /chat/completions by API key ID: {api_key_id}")
            else:
                # This case should ideally be blocked by middleware if auth is mandatory
                # and no key is provided or key is invalid.
                # However, if an exempt path somehow reaches here or middleware isn't strict,
                # or if auth can be optional.
                logger.info("Request to /chat/completions (auth enabled, but no API key found in state - check middleware logic for this path)")
        else:
            logger.info("Request to /chat/completions (authentication disabled)")

        trunc_request = await truncate_dict(request_body) # Use request_body here
        logger.info(f"Got request: {trunc_request}")
        content = await router.route_request(request_body) # Pass request_body to router
        return content
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"error": e.detail})
    except Exception as e:
        # Catch any other unexpected errors during request processing
        logger.error(f"Unexpected error in chat_completion: {e}", exc_info=True)
        return JSONResponse(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"error": "Internal Server Error"})


@app.get("/health")
async def health_check():
    router: Router = app.state.router
    try:
        content = await router.healthcheck()
        return {"status": content}
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"error": e.detail})


@app.get("/ok")
async def ok():
    return {"status": "ok"}


@app.get("/stats")
async def stats():
    router: Router = app.state.router
    try:
        content = await router.stats()
        return {"stats": content}
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"error": e.detail})


@app.get("/models")
@app.get("/v1/models")
async def models():
    router: Router = app.state.router
    try:
        content = await router.models()
        return content
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"error": e.detail})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
