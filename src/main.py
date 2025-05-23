import logging
from contextlib import asynccontextmanager

import coloredlogs
import requests
import uvicorn
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import load_config, load_bearer_tokens
from src.consts import PORT, LOG_LEVEL, EXTERNAL_HEALTHCHECK_URL
from src.router import Router
from src.utils import truncate_dict

logger = logging.getLogger(__name__)
coloredlogs.install(level=LOG_LEVEL, logger=logger)


# Dependency for token verification
async def verify_token(authorization: str = Header(None)):
    allowed_tokens = load_bearer_tokens()
    if not allowed_tokens:  # No tokens configured, auth disabled
        return True

    if not authorization:
        raise HTTPException(status_code=401, detail="Not authenticated")

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Not authenticated")

    token = parts[1]
    if token not in allowed_tokens:
        raise HTTPException(status_code=403, detail="Invalid token")
    return True


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

    app.state.router = Router(load_config())
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/chat/completions")
@app.post("/v1/chat/completions")
async def chat_completion(request: dict, _token_verified: bool = Depends(verify_token)):
    router: Router = app.state.router
    try:
        trunc_request = await truncate_dict(request)
        logger.info(f"Got request: {trunc_request}")
        content = await router.route_request(request)
        return content
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"error": e.detail})


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
async def stats(_token_verified: bool = Depends(verify_token)):
    router: Router = app.state.router
    try:
        content = await router.stats()
        return {"stats": content}
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"error": e.detail})


@app.get("/models")
@app.get("/v1/models")
async def models(_token_verified: bool = Depends(verify_token)):
    router: Router = app.state.router
    try:
        content = await router.models()
        return content
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"error": e.detail})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
