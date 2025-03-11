import logging
import random
from datetime import datetime, timedelta
from typing import List
from urllib.parse import urlparse

import coloredlogs
from fastapi import HTTPException
from httpx import AsyncClient, Response

from .config import ProviderConfig
from .consts import API_TIMEOUT_SECS, LOG_LEVEL


class Router:
    logger = logging.getLogger(__name__)
    coloredlogs.install(level=LOG_LEVEL, logger=logger)

    def __init__(self, providers: List[ProviderConfig]):
        # Group providers by priority and shuffle same-priority groups
        providers.sort(key=lambda x: x.priority)
        self.providers = []
        current_priority = None
        current_group = []

        for p in providers:
            if p.priority != current_priority:
                if current_group:
                    random.shuffle(current_group)
                    self.providers.extend(current_group)
                current_group = [p]
                current_priority = p.priority
            else:
                current_group.append(p)

        if current_group:
            random.shuffle(current_group)
            self.providers.extend(current_group)

        self.client = AsyncClient()

    async def route_request(self, request: dict):
        valid_providers = self._get_available_providers()

        if not valid_providers:
            raise HTTPException(
                status_code=429,
                detail="All providers are rate limited"
            )

        for provider in valid_providers:
            try:
                response = await self._make_request(provider, request)
                if response.status_code == 429:
                    self._handle_rate_limit(provider, response)
                    continue
                response_json = response.json()
                content = response_json.get("choices", [{}])[0].get("message", {}).get("content", None)
                if content is None:
                    self.logger.error(
                        f"Provider {provider.base_url} did not provide response in expected format: {response_json}"
                    )
                    continue
                self.logger.info(f"Got response from: {urlparse(provider.base_url).netloc}")
                return content
            except Exception as e:
                self.logger.error(f"Error with {provider.base_url}: {str(e)}")
                continue

        raise HTTPException(
            status_code=429,
            detail="All providers rate limited"
        )

    def _get_available_providers(self):
        now = datetime.now()
        return [
            p for p in self.providers
            if p.cooldown_until is None or p.cooldown_until < now
        ]

    async def _make_request(self, provider: ProviderConfig, request: dict):
        headers = {
            "Authorization": f"Bearer {provider.api_key}",
            "Content-Type": "application/json"
        }
        payload = {**request, "model": provider.model_name}

        try:
            api_url = f"{provider.base_url}/chat/completions"
            response = await self.client.post(
                api_url,
                json=payload,
                headers=headers,
                timeout=API_TIMEOUT_SECS
            )
            return response
        except Exception as e:
            self.logger.error(f"Request failed to {provider.base_url}: {str(e)}")
            raise

    def _handle_rate_limit(self, provider: ProviderConfig, response: Response):
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                cooldown = int(retry_after)
            except ValueError:
                cooldown = 60  # Default to 60 seconds
            provider.cooldown_until = datetime.now() + timedelta(seconds=cooldown)
        else:
            provider.cooldown_until = datetime.now() + timedelta(seconds=60)

        self.logger.warning(
            f"Provider {provider.base_url} rate limited. Cooldown until {provider.cooldown_until}"
        )
