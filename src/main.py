import logging
from contextlib import asynccontextmanager

import coloredlogs
import uvicorn
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import load_config
from src.consts import PORT, LOG_LEVEL
from src.router import Router
from src.utils import truncate_dict

logger = logging.getLogger(__name__)
coloredlogs.install(level=LOG_LEVEL, logger=logger)


@asynccontextmanager
async def lifespan(app: FastAPI):
    scheduler = BackgroundScheduler()
    scheduler.add_job(health_check, "interval", minutes=15)
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


@app.post("/v1/chat/completions")
async def chat_completion(request: dict):
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
async def stats():
    router: Router = app.state.router
    try:
        content = await router.stats()
        return {"stats": content}
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"error": e.detail})


@app.get("/models")
async def models():
    router: Router = app.state.router
    try:
        content = await router.models()
        return {"models": content}
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"error": e.detail})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
