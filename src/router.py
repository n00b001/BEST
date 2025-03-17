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
        self.base_cooldowns = {}  # {base_url: cooldown_end_time}
        self.model_cooldowns = {}  # {(base_url, model_name): cooldown_end_time}
        self.base_failure_counts = {}  # {base_url: consecutive_failure_count}
        self.model_failure_counts = {}  # {(base_url, model_name): consecutive_failure_count}

    async def healthcheck(self):
        response = await self.client.get(url=HEALTHCHECK_URL)
        return response

    async def route_request(self, request: dict):
        valid_providers = self._get_available_providers()

        if not valid_providers:
            raise HTTPException(status_code=429, detail="All providers are rate limited")

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
                # Reset failure counts on successful request
                self.base_failure_counts.pop(provider.base_url, None)
                model_key = (provider.base_url, provider.model_name)
                self.model_failure_counts.pop(model_key, None)
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

    def _handle_rate_limit(self, provider: ProviderConfig, response: Union[Response, None], cooldown):
        if response is not None:
            retry_after = response.headers.get("Retry-After")
        else:
            retry_after = None

        is_rate_limit = response is not None and response.status_code == 429

        # Determine base cooldown parameters
        if is_rate_limit:
            base_url = provider.base_url
            current_failures = self.base_failure_counts.get(base_url, 0) + 1
            self.base_failure_counts[base_url] = current_failures

            # Calculate cooldown duration
            if retry_after:
                try:
                    base_cooldown = int(retry_after)
                except ValueError:
                    base_cooldown = DEFAULT_COOLDOWN_SECONDS
            else:
                base_cooldown = DEFAULT_COOLDOWN_SECONDS * (2 ** (current_failures - 1))

            # Apply base-level cooldown
            cooldown_time = datetime.now() + timedelta(seconds=base_cooldown)
            self.base_cooldowns[base_url] = cooldown_time
            self.logger.warning(f"Base URL {base_url} cooldown until {cooldown_time} (failures: {current_failures})")
        else:
            model_key = (provider.base_url, provider.model_name)
            current_failures = self.model_failure_counts.get(model_key, 0) + 1
            self.model_failure_counts[model_key] = current_failures

            # Determine default cooldown based on error type
            if response and response.status_code // 100 == 4:
                default_cooldown = BAD_REQUEST_COOLDOWN_SECONDS
            else:
                default_cooldown = DEFAULT_COOLDOWN_SECONDS

            # Calculate cooldown duration
            if retry_after:
                try:
                    model_cooldown = int(retry_after)
                except ValueError:
                    model_cooldown = default_cooldown
            else:
                model_cooldown = default_cooldown * (2 ** (current_failures - 1))

            # Apply model-level cooldown
            cooldown_time = datetime.now() + timedelta(seconds=model_cooldown)
            self.model_cooldowns[model_key] = cooldown_time
            self.logger.warning(
                f"Model {provider.model_name} cooldown until {cooldown_time} (failures: {current_failures})")

