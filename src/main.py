import builtins
import logging
import textwrap
from contextlib import asynccontextmanager

import coloredlogs
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import load_config
from src.consts import PORT, LOG_LEVEL, MAX_REQUEST_CHAR_COUNT_FOR_LOG
from src.router import Router

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
async def chat_completion(request: dict):
    router: Router = app.state.router
    try:
        trunc_request = await get_truncated_dict(request)
        logger.info(f"Got request: {trunc_request}")
        content = await router.route_request(request)
        return content
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"error": e.detail})


async def get_truncated_dict(_dict):
    trunc_request = {}
    for k, v in _dict.items():
        await switch_case_for_type(k, trunc_request, v)

    return trunc_request


async def switch_case_for_type(k, trunc_request, v):
    match type(v):
        case builtins.str:
            trunc_request[k] = await truncate_str(v)
        case builtins.list:
            for item in v:
                await switch_case_for_type(k, trunc_request, item)
        case builtins.dict:
            trunc_request[k] = await get_truncated_dict(v)


async def truncate_str(v):
    return textwrap.shorten(
        v, width=MAX_REQUEST_CHAR_COUNT_FOR_LOG, placeholder="..."
    )


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
