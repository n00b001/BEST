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
    EXTERNAL_HEALTHCHECK_URL, NON_PROJECT_HEALTHCHECK_URL,
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
        response = await self.client.get(url=NON_PROJECT_HEALTHCHECK_URL)
        response.raise_for_status()
        return response.is_success

    async def models(self):
        unique_models = {}
        # Collect provider models
        for provider in self.providers:
            model_id = provider.model_name
            if model_id not in unique_models:
                unique_models[model_id] = {
                    "id": model_id,
                    "object": "model",
                    "created": 1686935002,  # Example timestamp
                    "owned_by": urlparse(provider.base_url).netloc
                }
        # Add priority-auto model
        unique_models["priority-auto"] = {
            "id": "priority-auto",
            "object": "model",
            "created": 1686935002,
            "owned_by": "proxy-system"
        }
        models_list = list(unique_models.values())
        return {
            "object": "list",
            "data": models_list
        }

    async def stats(self):
        now = datetime.now()
        providers_stats = []
        total_stats = {
            "total_successes": 0,
            "total_failures": 0,
            "total_input_tokens": 0,
            "total_generated_tokens": 0,
            "all_latencies": [],
            "available_count": 0
        }

        for provider in self.providers:
            provider_stats = self._process_provider_stats(provider, now, total_stats)
            providers_stats.append(provider_stats)

        overall_stats = self._calculate_overall_stats(total_stats)

        return {
            "providers": providers_stats,
            "total_providers": len(self.providers),
            "available_providers": total_stats["available_count"],
            "timestamp": now.isoformat(),
            "overall": overall_stats
        }

    def _process_provider_stats(self, provider, now, total_stats):
        base_url = provider.base_url
        model_key = (base_url, provider.model_name)

        base_remaining, model_remaining = self._calculate_cooldowns(base_url, model_key, now)
        is_available = base_remaining <= 0 and model_remaining <= 0
        stats = self.model_stats[model_key]

        if is_available:
            total_stats["available_count"] += 1

        # Calculate metrics
        latency_metrics = self._get_latency_metrics(stats["latencies"])
        failure_rate = self._calculate_failure_rate(stats["successes"], stats["failures"])

        # Update aggregates
        self._update_total_stats(total_stats, stats, latency_metrics)

        return {
            "priority": provider.priority,
            "base_url": base_url,
            "model_name": provider.model_name,
            "base_cooldown_remaining": base_remaining,
            "model_cooldown_remaining": model_remaining,
            "base_failures": self.base_failure_counts.get(base_url, 0),
            "model_failures": self.model_failure_counts.get(model_key, 0),
            "is_available": is_available,
            "successful_calls": stats["successes"],
            "failed_calls": stats["failures"],
            "failure_rate": round(failure_rate, 2),
            "input_tokens": stats["input_tokens"],
            "generated_tokens": stats["generated_tokens"],
            **latency_metrics
        }

    def _calculate_cooldowns(self, base_url, model_key, now):
        def get_remaining(cooldown_dict, key):
            end_time = cooldown_dict.get(key)
            return max(0, (end_time - now).total_seconds()) if end_time and end_time > now else 0

        return (
            get_remaining(self.base_cooldowns, base_url),
            get_remaining(self.model_cooldowns, model_key)
        )

    def _get_latency_metrics(self, latencies):
        metrics = {
            "min_latency": 0.0,
            "max_latency": 0.0,
            "mean_latency": 0.0,
            "latencies": latencies  # Add the actual latencies list here
        }

        if latencies:
            metrics.update({
                "min_latency": round(min(latencies), 3),
                "max_latency": round(max(latencies), 3),
                "mean_latency": round(sum(latencies) / len(latencies), 3)
            })

        return metrics

    def _calculate_failure_rate(self, successes, failures):
        total = successes + failures
        return (failures / total * 100) if total > 0 else 0.0

    def _update_total_stats(self, total_stats, stats, latency_metrics):
        total_stats["total_successes"] += stats["successes"]
        total_stats["total_failures"] += stats["failures"]
        total_stats["total_input_tokens"] += stats["input_tokens"]
        total_stats["total_generated_tokens"] += stats["generated_tokens"]
        # Use the latencies list from metrics
        total_stats["all_latencies"].extend(latency_metrics["latencies"])

    def _calculate_overall_stats(self, total_stats):
        latency_metrics = self._get_latency_metrics(total_stats["all_latencies"])

        return {
            "successful_calls": total_stats["total_successes"],
            "failed_calls": total_stats["total_failures"],
            "failure_rate": round(self._calculate_failure_rate(
                total_stats["total_successes"],
                total_stats["total_failures"]
            ), 2),
            "input_tokens": total_stats["total_input_tokens"],
            "generated_tokens": total_stats["total_generated_tokens"],
            **latency_metrics
        }

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

    async def route_request(self, request: dict):
        model_param = request.get("model", "priority-auto")

        if model_param == "priority-auto":
            providers_to_try = sorted(
                self._get_available_providers(),
                key=lambda p: p.priority["overall_score"]
            )
        else:
            model_names = [name.strip() for name in model_param.split(",")]
            providers_to_try = []
            for model_name in model_names:
                model_providers = [
                    p for p in self._get_available_providers()
                    if model_name in p.model_name
                ]
                sorted_providers = sorted(
                    model_providers,
                    key=lambda p: p.priority["overall_score"]
                )
                providers_to_try.extend(sorted_providers)

        if not providers_to_try:
            raise HTTPException(status_code=429, detail="All providers are rate limited")

        for provider in providers_to_try:
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
