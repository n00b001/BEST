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
        self.base_cooldowns = {}
        self.model_cooldowns = {}

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

        raise HTTPException(status_code=429, detail="All providers rate limited")

    def _get_available_providers(self):
        now = datetime.now()
        available = []
        for provider in self.providers:
            base_cooldown_end = self.base_cooldowns.get(provider.base_url)
            if base_cooldown_end and base_cooldown_end > now:
                continue  # Base URL is on cooldown

            model_key = (provider.base_url, provider.model_name)
            model_cooldown_end = self.model_cooldowns.get(model_key)
            if model_cooldown_end and model_cooldown_end > now:
                continue  # Model is on cooldown

            available.append(provider)
        return available

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
            self.logger.debug(f"Request failed to {provider.base_url}: {str(e)}")
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
                pass  # Keep the original cooldown value if Retry-After is invalid

        is_rate_limit = False
        if response is not None and response.status_code == 429:
            is_rate_limit = True

        cooldown_time = datetime.now() + timedelta(seconds=cooldown)
        if is_rate_limit:
            self.base_cooldowns[provider.base_url] = cooldown_time
            self.logger.warning(f"Base URL {provider.base_url} cooldown until {cooldown_time}")
        else:
            model_key = (provider.base_url, provider.model_name)
            self.model_cooldowns[model_key] = cooldown_time
            self.logger.warning(f"Model {provider.model_name} cooldown until {cooldown_time}")
