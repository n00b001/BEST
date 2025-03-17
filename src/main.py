import logging
from contextlib import asynccontextmanager

import coloredlogs
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import load_config
from src.consts import PORT, LOG_LEVEL
from src.router import Router
from src.utils import truncate_dict

logger = logging.getLogger(__name__)
coloredlogs.install(level=LOG_LEVEL, logger=logger)


from typing import AsyncGenerator

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    app.state.router = Router(load_config())
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


from fastapi import Request

@app.post("/v1/chat/completions")
async def chat_completion(request: Request) -> JSONResponse:
    router: Router = app.state.router
    try:
        request_dict = await request.json()
        trunc_request = await truncate_dict(request_dict)
        logger.info(f"Got request: {trunc_request}")
        content = await router.route_request(request_dict)
        return content
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"error": e.detail})


from fastapi.responses import Response

@app.get("/health")
async def health_check() -> Response:
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
