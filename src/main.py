import logging
from contextlib import asynccontextmanager

import coloredlogs
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import load_config
from src.consts import PORT, LOG_LEVEL
from src.router import Router
from src.utils import truncate_dict
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
security = HTTPBearer()

async def get_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    # Validate the token against the secret
    if token != "your_secret_token_here":
        raise HTTPException(status_code=401, detail="Invalid token")
    return token

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
async def chat_completion(request: Request, token: str = Depends(get_token)):  # Change to Request
    router: Router = app.state.router
    try:
        trunc_request = await truncate_dict(await request.json())  # Get request body
        logger.info(f"Got request: {trunc_request}")
        content = await router.route_request(request)  # Pass the request object
        return content
    except HTTPException as e:
        return JSONResponse(status_code=401, content={"error": "Unauthorized"})  # Correct error format


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
