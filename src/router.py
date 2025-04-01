import logging
from collections import defaultdict
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Any
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
        self.base_cooldowns: Dict[str, datetime] = {}
        self.model_cooldowns: Dict[Tuple[str, str], datetime] = {}
        self.base_failure_counts: Dict[str, int] = {}
        self.model_failure_counts: Dict[Tuple[str, str], int] = {}
        self.model_stats: defaultdict[Any, dict[str, Any]] = defaultdict(
            lambda: {"successes": 0, "failures": 0, "input_tokens": 0, "generated_tokens": 0, "latencies": []}
        )

    async def healthcheck(self):
        response = await self.client.get(url=HEALTHCHECK_URL)
        response.raise_for_status()
        return response.is_success

    async def models(self):
        # Collect model information from each provider
        models_list = [{"name": provider.model_name, "priority": provider.priority} for provider in self.providers]
        # Sort the models by their priority (ascending order)
        models_list.sort(key=lambda x: x["priority"]["overall_score"])
        return models_list

    async def stats(self):
        now = datetime.now()
        providers_stats = []
        available_count = 0

        # Initialize overall stats
        total_successes = 0
        total_failures = 0
        total_input_tokens = 0
        total_generated_tokens = 0
        all_latencies = []

        for provider in self.providers:
            base_url = provider.base_url
            model_name = provider.model_name
            model_key = (base_url, model_name)

            # Cooldown calculations
            base_cooldown_end = self.base_cooldowns.get(base_url)
            base_remaining = (
                max(0, (base_cooldown_end - now).total_seconds())
                if base_cooldown_end and base_cooldown_end > now
                else 0
            )

            model_cooldown_end = self.model_cooldowns.get(model_key)
            model_remaining = (
                max(0, (model_cooldown_end - now).total_seconds())
                if model_cooldown_end and model_cooldown_end > now
                else 0
            )

            is_available = base_remaining <= 0 and model_remaining <= 0
            if is_available:
                available_count += 1

            # Get model-specific stats
            stats = self.model_stats[model_key]
            successes = stats["successes"]
            failures = stats["failures"]
            total_calls = successes + failures
            failure_rate = (failures / total_calls * 100) if total_calls > 0 else 0.0

            latencies = stats["latencies"]
            min_latency = min(latencies) if latencies else 0
            max_latency = max(latencies) if latencies else 0
            mean_latency = (sum(latencies) / len(latencies)) if latencies else 0

            # Update overall aggregates
            total_successes += successes
            total_failures += failures
            total_input_tokens += stats["input_tokens"]
            total_generated_tokens += stats["generated_tokens"]
            all_latencies.extend(latencies)

            # Build provider stats entry
            provider_stats = {
                "base_url": base_url,
                "model_name": model_name,
                "base_cooldown_remaining": base_remaining,
                "model_cooldown_remaining": model_remaining,
                "base_failures": self.base_failure_counts.get(base_url, 0),
                "model_failures": self.model_failure_counts.get(model_key, 0),
                "is_available": is_available,
                "successful_calls": successes,
                "failed_calls": failures,
                "failure_rate": round(failure_rate, 2),
                "input_tokens": stats["input_tokens"],
                "generated_tokens": stats["generated_tokens"],
                "min_latency": round(min_latency, 3),
                "max_latency": round(max_latency, 3),
                "mean_latency": round(mean_latency, 3),
                "priority": provider.priority,
            }
            providers_stats.append(provider_stats)

        # Calculate overall metrics
        total_calls = total_successes + total_failures
        overall_failure_rate = (total_failures / total_calls * 100) if total_calls > 0 else 0.0
        overall_min_latency = min(all_latencies) if all_latencies else 0
        overall_max_latency = max(all_latencies) if all_latencies else 0
        overall_mean_latency = (sum(all_latencies) / len(all_latencies)) if all_latencies else 0

        stats_dict = {
            "providers": providers_stats,
            "total_providers": len(self.providers),
            "available_providers": available_count,
            "timestamp": now.isoformat(),
            "overall": {
                "successful_calls": total_successes,
                "failed_calls": total_failures,
                "failure_rate": round(overall_failure_rate, 2),
                "input_tokens": total_input_tokens,
                "generated_tokens": total_generated_tokens,
                "min_latency": round(overall_min_latency, 3),
                "max_latency": round(overall_max_latency, 3),
                "mean_latency": round(overall_mean_latency, 3),
            },
        }

        return stats_dict

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

    def _process_successful_response(self, provider: ProviderConfig, response: Response, latency: float):
        response_json = response.json()
        content = response_json.get("choices", [{}])[0].get("message", {}).get("content", None)

        if content is None:
            self.logger.error(f"Provider {provider.base_url} bad format: {response_json}")
            raise ValueError("Invalid response format")

        # Update success metrics
        model_key = (provider.base_url, provider.model_name)
        self.model_stats[model_key]["successes"] += 1
        usage = response_json.get("usage", {})
        self.model_stats[model_key]["input_tokens"] += usage.get("prompt_tokens", 0)
        self.model_stats[model_key]["generated_tokens"] += usage.get("completion_tokens", 0)
        self.model_stats[model_key]["latencies"].append(latency)

        self._reset_failure_counts(provider)
        self.logger.info(f"Got response from: {urlparse(provider.base_url).netloc}")
        return response_json

    async def _handle_provider_error(
        self, provider: ProviderConfig, error: Exception, response: Response | None, latency: float
    ):
        model_key = (provider.base_url, provider.model_name)
        self.model_stats[model_key]["failures"] += 1
        self.model_stats[model_key]["latencies"].append(latency)

        if response is not None:
            await self._handle_response_error(provider, response, error)
        else:
            self.logger.error(f"Error with {provider.base_url}: {str(error)}")
            self._handle_rate_limit(provider, None, DEFAULT_COOLDOWN_SECONDS)

    async def route_request(self, request: dict, token: str | None = None):
        valid_providers = self._get_available_providers()
def _validate_token(self, token: str | None):
        secret_token = os.getenv("API_SECRET_TOKEN", "TESTING")
        if not token or token != secret_token:
            raise HTTPException(status_code=401, detail="Unauthorized")
def _validate_token(self, token: str | None):
        secret_token = os.getenv("API_SECRET_TOKEN", "TESTING")
        if not token or token != secret_token:
            raise HTTPException(status_code=401, detail="Unauthorized")
def _validate_token(self, token: str | None):
        secret_token = os.getenv("API_SECRET_TOKEN", "TESTING")
        if not token or token != secret_token:
            raise HTTPException(status_code=401, detail="Unauthorized")
def _validate_token(self, token: str | None):
        if not token or token != "TESTING":
            raise HTTPException(status_code=401, detail="Unauthorized")
def _validate_token(self, token: str | None):
        if not token or token != "TESTING":
            raise HTTPException(status_code=401, detail="Unauthorized")
        if not valid_providers:
            raise HTTPException(status_code=429, detail="All providers are rate limited")

        for provider in valid_providers:
            start_time = datetime.now()
            try:
                response = await self._make_request(provider, request)
                latency = (datetime.now() - start_time).total_seconds()
                response.raise_for_status()
                return self._process_successful_response(provider, response, latency)
            except Exception as e:
                latency = (datetime.now() - start_time).total_seconds()
                await self._handle_provider_error(provider, e, response, latency)

        raise HTTPException(status_code=429, detail="All providers rate limited")

    def _reset_failure_counts(self, provider: ProviderConfig):
        self.base_failure_counts.pop(provider.base_url, None)
        model_key = (provider.base_url, provider.model_name)
        self.model_failure_counts.pop(model_key, None)

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
