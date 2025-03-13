import logging
from datetime import datetime, timedelta
from typing import List, Union
from urllib.parse import urlparse

import coloredlogs
from fastapi import HTTPException
from httpx import AsyncClient, Response

from .config import ProviderConfig
from .consts import (
    API_TIMEOUT_SECS,
    LOG_LEVEL,
    DEFAULT_COOLDOWN_SECONDS,
    BAD_REQUEST_COOLDOWN_SECONDS,
    HEALTHCHECK_URL,
)


class Router:
    logger = logging.getLogger(__name__)
    coloredlogs.install(level=LOG_LEVEL, logger=logger)

    def __init__(self, providers: List[ProviderConfig]):
        self.providers = providers
        self.client = AsyncClient()

    async def healthcheck(self):
        response = await self.client.get(url=HEALTHCHECK_URL)
        return response

    async def route_request(self, request: dict):
        valid_providers = self._get_available_providers()

        if not valid_providers:
            raise HTTPException(status_code=429, detail="All providers are rate limited")

        # todo the randomization between providers of the same priority should be done at call time
        for provider in valid_providers:
            response = None
            try:
                response = await self._make_request(provider, request)
                response.raise_for_status()
                response_json = response.json()
                content = response_json.get("choices", [{}])[0].get("message", {}).get("content", None)
                if content is None:
                    self.logger.error(
                        f"Provider {provider.base_url} did not provide response in expected format: {response_json}"
                    )
                    continue
                self.logger.info(f"Got response from: {urlparse(provider.base_url).netloc}")
                return response_json
            except Exception as e:
                if response is not None:
                    if response.status_code == 429:
                        self.logger.error(
                            f"Error with {provider.base_url}, "
                            f"Rate limited.., "
                            f"error message: {response.text}, "
                            f"{str(e)}"
                        )
                        self._handle_rate_limit(provider, response, DEFAULT_COOLDOWN_SECONDS)
                    elif int(response.status_code / 100) == 4:
                        self.logger.error(
                            f"Error with {provider.base_url}, "
                            f"status code: {response.status_code}, "
                            f"error message: {response.text}, "
                            f"{str(e)}"
                        )
                        self._handle_rate_limit(provider, response, BAD_REQUEST_COOLDOWN_SECONDS)
                    else:
                        self.logger.error(
                            f"Error with {provider.base_url}, status code: {response.status_code}: {str(e)}"
                        )
                        self._handle_rate_limit(provider, response, DEFAULT_COOLDOWN_SECONDS)
                else:
                    self.logger.error(f"Error with {provider.base_url}: {str(e)}")
                    self._handle_rate_limit(provider, response, DEFAULT_COOLDOWN_SECONDS)
                continue

        if not request.get("token"):
    raise HTTPException(status_code=401, detail="Invalid token")
token = request.get("token")
secret_key = os.getenv("SECRET_KEY")
if token != secret_key:
    raise HTTPException(status_code=401, detail="Invalid token")
if not request.get("token"):
        raise HTTPException(status_code=401, detail="Invalid token")
    token = request.get("token")
    secret_key = os.getenv("SECRET_KEY")
    if token != secret_key:
        raise HTTPException(status_code=401, detail="Invalid token")

    # ... (rest of the original code) ...
    if not request.get("token"):
        raise HTTPException(status_code=401, detail="Missing token")
    token = request.get("token")
    secret_key = os.getenv("SECRET_KEY")
    if token != secret_key:
        raise HTTPException(status_code=401, detail="Invalid token")

    valid_providers = self._get_available_providers()
    if not valid_providers:
        raise HTTPException(status_code=429, detail="All providers are rate limited")

    # ... (rest of the original code) ...
    if not request.get("token"):
        raise HTTPException(status_code=401, detail="Missing token")
    token = request.get("token")
    secret_key = os.getenv("SECRET_KEY")
    if token != secret_key:
        raise HTTPException(status_code=401, detail="Invalid token")

    valid_providers = self._get_available_providers()
    if not valid_providers:
        raise HTTPException(status_code=429, detail="All providers are rate limited")

    for provider in valid_providers:
        try:
            response = await self._make_request(provider, request)
            response.raise_for_status()
            # ... (rest of the original code) ...
        except HTTPException as e:
            if e.status_code == 429:
                self._handle_rate_limit(provider, response, DEFAULT_COOLDOWN_SECONDS)
            elif e.status_code in (400, 401, 404, 500):
                self._handle_rate_limit(provider, response, BAD_REQUEST_COOLDOWN_SECONDS)
            else:
                self._handle_rate_limit(provider, response, DEFAULT_COOLDOWN_SECONDS)
            continue  # Move to the next provider
        except Exception as e:
            self.logger.error(f"Error with {provider.base_url}: {str(e)}")
            self._handle_rate_limit(provider, None, DEFAULT_COOLDOWN_SECONDS)
            continue  # Move to the next provider

    raise HTTPException(status_code=429, detail="All providers are rate limited")  # Should only reach here if all providers fail

    def _get_available_providers(self):
        now = datetime.now()
        return [p for p in self.providers if p.cooldown_until is None or p.cooldown_until < now]

    async def _make_request(self, provider: ProviderConfig, request: dict):
        headers = {
            "Authorization": f"Bearer {provider.api_key}",
            "Content-Type": "application/json",
        }
        payload = {**request, "model": provider.model_name}

        try:
            api_url = f"{provider.base_url}/chat/completions"
            response = await self.client.post(api_url, json=payload, headers=headers, timeout=API_TIMEOUT_SECS)
            return response
        except Exception as e:
            self.logger.error(f"Request failed to {provider.base_url}: {str(e)}")
            raise

    def _handle_rate_limit(self, provider: ProviderConfig, response: Union[Response, None], cooldown):
        if response is not None:
            retry_after = response.headers.get("Retry-After")
        else:
            retry_after = None

        if retry_after is not None:
            try:
                cooldown = int(retry_after)
            except ValueError:
                cooldown = DEFAULT_COOLDOWN_SECONDS  # Default to 60 seconds
        provider.cooldown_until = datetime.now() + timedelta(seconds=cooldown)

        self.logger.warning(f"Provider {provider.base_url} rate limited. Cooldown until {provider.cooldown_until}")
