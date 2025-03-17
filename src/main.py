import logging
from contextlib import asynccontextmanager

import coloredlogs
import uvicorn
from fastapi import FastAPI, HTTPException, Request  # Import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import load_config
from consts import PORT, LOG_LEVEL
from router import Router
from utils import truncate_dict

logger = logging.getLogger(__name__)
coloredlogs.install(level=LOG_LEVEL, logger=logger)


@asynccontextmanager
async def lifespan(app: FastAPI):
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
async def chat_completion(request: Request):  # Change to Request
    router: Router = app.state.router
    try:
        trunc_request = await truncate_dict(await request.json())  # Get request body
        logger.info(f"Got request: {trunc_request}")
        content = await router.route_request(request)  # Pass the request object
        return content
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"error": "Unauthorized"})  # Correct error format


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
