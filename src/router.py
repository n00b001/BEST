import logging
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from urllib.parse import urlparse

import coloredlogs
from fastapi import HTTPException
from httpx import AsyncClient, Response

from config import ProviderConfig
from consts import (
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
        self.base_cooldowns: Dict[str, datetime] = {}
        self.model_cooldowns: Dict[Tuple[str, str], datetime] = {}
        self.base_failure_counts: Dict[str, int] = {}
        self.model_failure_counts: Dict[Tuple[str, str], int] = {}

    async def healthcheck(self):
        response = await self.client.get(url=HEALTHCHECK_URL)
        return response

    def _get_available_providers(self):
        now = datetime.now()
        available = []
        for provider in self.providers:
            # Check base URL cooldown
            base_cooldown_end = self.base_cooldowns.get(provider.base_url)
            if base_cooldown_end and base_cooldown_end > now:
                continue

            # Check model-specific cooldown
            model_key = (provider.base_url, provider.model_name)
            model_cooldown_end = self.model_cooldowns.get(model_key)
            if model_cooldown_end and model_cooldown_end > now:
                continue

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

    async def route_request(self, request: Request):  # Change to Request
        import os
    auth_header = request.headers.get("Authorization")  # Get from headers
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")

    token = auth_header.split(" ")[1]
    api_secret = os.environ.get("API_SECRET")
    if not api_secret or token != api_secret:
        raise HTTPException(status_code=401, detail="Unauthorized")

    valid_providers = self._get_available_providers()
    if not valid_providers:
        raise HTTPException(status_code=429, detail="All providers are rate limited")

    for provider in valid_providers:
        try:
            response = await self._make_request(provider, await request.json())
            response.raise_for_status()
            return self._process_successful_response(provider, response)
        except Exception as e:
            await self._handle_provider_error(provider, e, response)

    raise HTTPException(status_code=429, detail="All providers rate limited")

    def _process_successful_response(self, provider: ProviderConfig, response: Response):
        response_json = response.json()
        content = response_json.get("choices", [{}])[0].get("message", {}).get("content", None)

        if content is None:
            self.logger.error(f"Provider {provider.base_url} bad format: {response_json}")
            raise ValueError("Invalid response format")

        self._reset_failure_counts(provider)
        self.logger.info(f"Got response from: {urlparse(provider.base_url).netloc}")
        return response_json

    def _reset_failure_counts(self, provider: ProviderConfig):
        self.base_failure_counts.pop(provider.base_url, None)
        model_key = (provider.base_url, provider.model_name)
        self.model_failure_counts.pop(model_key, None)

    async def _handle_provider_error(self, provider: ProviderConfig, error: Exception, response: Response | None):
        if response is not None:
            await self._handle_response_error(provider, response, error)
        else:
            self.logger.error(f"Error with {provider.base_url}: {str(error)}")
            self._handle_rate_limit(provider, None, DEFAULT_COOLDOWN_SECONDS)

    async def _handle_response_error(self, provider: ProviderConfig, response: Response, error: Exception):
        status_code = response.status_code
        error_msg = f"{provider.base_url} status {status_code}: {response.text} {str(error)}"

        if status_code == 429:
            self.logger.error(f"Rate limited: {error_msg}")
            self._handle_rate_limit(provider, response, DEFAULT_COOLDOWN_SECONDS)
        elif status_code // 100 == 4:
            self.logger.error(f"Client error: {error_msg}")
            self._handle_rate_limit(provider, response, BAD_REQUEST_COOLDOWN_SECONDS)
        else:
            self.logger.error(f"Server error: {error_msg}")
            self._handle_rate_limit(provider, response, DEFAULT_COOLDOWN_SECONDS)

    def _handle_rate_limit(self, provider: ProviderConfig, response: Response | None, default_cooldown: int):
        retry_after = self._get_retry_after(response)
        is_rate_limit = response and response.status_code == 429

        if is_rate_limit:
            self._apply_base_cooldown(provider, retry_after)
        else:
            self._apply_model_cooldown(provider, response, retry_after, default_cooldown)

    def _get_retry_after(self, response: Response | None) -> int | None:
        return int(response.headers["Retry-After"]) if response and "Retry-After" in response.headers else None

    def _apply_base_cooldown(self, provider: ProviderConfig, retry_after: int | None):
        base_url = provider.base_url
        current_failures = self.base_failure_counts.get(base_url, 0) + 1
        self.base_failure_counts[base_url] = current_failures

        cooldown = self._calculate_cooldown(retry_after, DEFAULT_COOLDOWN_SECONDS, current_failures)
        cooldown_time = datetime.now() + timedelta(seconds=cooldown)
        self.base_cooldowns[base_url] = cooldown_time
        self.logger.warning(f"Base {base_url} cooldown until {cooldown_time} (failures: {current_failures})")

    def _apply_model_cooldown(
        self, provider: ProviderConfig, response: Response | None, retry_after: int | None, default_cooldown: int
    ):
        model_key = (provider.base_url, provider.model_name)
        current_failures = self.model_failure_counts.get(model_key, 0) + 1
        self.model_failure_counts[model_key] = current_failures

        base_cooldown = (
            BAD_REQUEST_COOLDOWN_SECONDS if response and response.status_code // 100 == 4 else default_cooldown
        )
        cooldown = self._calculate_cooldown(retry_after, base_cooldown, current_failures)

        cooldown_time = datetime.now() + timedelta(seconds=cooldown)
        self.model_cooldowns[model_key] = cooldown_time
        self.logger.warning(
            f"Model {provider.model_name} cooldown until {cooldown_time} (failures: {current_failures})"
        )

    def _calculate_cooldown(self, retry_after: int | None, base_cooldown: int, failures: int) -> int:
        if retry_after is not None:
            return retry_after
        return base_cooldown * (2 ** (failures - 1))
