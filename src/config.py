import logging
import os
import re
import time
from typing import List
from urllib.parse import urlparse

import coloredlogs
import requests
import ruamel.yaml
from dotenv import load_dotenv
from pydantic import BaseModel
from ruamel.yaml.scalarstring import DoubleQuotedScalarString

from src.consts import (
    LOG_LEVEL,
    DOT_ENV_FILENAME,
    DEFAULT_CONFIG_LOCATION,
    META_PROVIDERS_CONFIG_FILENAME,
    GENERATED_PROVIDER_CONFIG_STALE_TIME_SECS,
    MODEL_PARAM_SCORE_SCALAR,
    MODEL_CTX_SCORE_SCALAR,
    MODEL_ADJUSTMENTS_FILENAME,
)
from src.model_score import aggregate_model_scores
from src.utils import truncate_dict

logger = logging.getLogger(__name__)
coloredlogs.install(level=LOG_LEVEL, logger=logger)

yaml = ruamel.yaml.YAML()
yaml.preserve_quotes = True
yaml.explicit_start = True
yaml.indent(sequence=4, offset=2)


class ProviderConfig(BaseModel):
    api_key: str
    base_url: str
    model_name: str
    priority: float


def _load_model_adjustments() -> dict:
    try:
        with open(MODEL_ADJUSTMENTS_FILENAME, "r") as f:
            config = yaml.load(f) or {}
        return config.get("adjustments", {})
    except FileNotFoundError:
        logger.warning(f"No model adjustments found at {MODEL_ADJUSTMENTS_FILENAME}")
        return {}
    except Exception as e:
        logger.error(f"Error loading model adjustments: {e}")
        return {}


def generate_providers(config_path):
    with open(META_PROVIDERS_CONFIG_FILENAME, "r") as f:
        meta_providers = yaml.load(f)["providers"]

    model_adjustments = _load_model_adjustments()
    output = {"providers": []}

    for provider in meta_providers:
        provider_models = _process_provider(provider, model_adjustments)
        if provider_models:
            output["providers"].extend(provider_models)

    _write_output_config(config_path, output)
    logger.info("Providers configuration generated successfully")


def _process_provider(provider: dict, model_adjustments: dict) -> list[dict]:
    api_key = os.getenv(provider["api_key_env_var"])
    if not api_key:
        logger.warning(f"Skipping {provider['base_url']} - API key missing")
        return []

    try:
        model_data_list = _fetch_provider_models(provider, api_key)
        return _process_provider_models(provider, model_data_list, model_adjustments)
    except Exception as e:
        logger.error(f"Error processing {provider['base_url']}: {e}")
        return []


def _fetch_provider_models(provider: dict, api_key: str) -> list:
    model_url = provider.get("model_url", provider["base_url"])
    response = requests.get(
        f"{model_url}/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    response.raise_for_status()
    models_data = response.json()
    if isinstance(models_data, list):
        return models_data

    model_list = models_data.get("data", models_data.get("models", []))
    return model_list


def _process_provider_models(provider: dict, model_data_list: list, model_adjustments: dict) -> list[dict]:
    processed_models = []
    for model in model_data_list:
        model_name = model.get("id", model.get("name", None))
        if model_name is None:
            logger.warning(f"Unable to get model name, skipping: {truncate_dict(model)}")
            continue
        context_length = model.get("context_length", 0)
        priority = _calculate_model_priority(provider, model_name, context_length, model_adjustments)
        processed_models.append(_create_model_config(provider, model_name, priority))
    logger.info(f"Got {len(processed_models)} models for: {urlparse(provider['base_url']).netloc}")
    return processed_models


def _calculate_model_priority(provider: dict, model_name: str, context_length: int, model_adjustments: dict) -> float:
    model_score = aggregate_model_scores(model_name)

    # Parameter-based scoring
    if match := re.search(r"-(\d+)b", model_name):
        model_score += int(match.group(1)) * MODEL_PARAM_SCORE_SCALAR

    # Mixtral-style model scoring
    if x_match := re.search(r"(\d+)x(\d+)b", model_name):
        model_score += int(x_match.group(1)) * int(x_match.group(2)) * MODEL_PARAM_SCORE_SCALAR

    # Context length scoring
    model_score += context_length * MODEL_CTX_SCORE_SCALAR

    # Apply global model adjustments with fuzzy matching
    normalized_model = re.sub(r"\W+", "", model_name).lower()
    matches = []

    for adjust_key, adjust_value in model_adjustments.items():
        normalized_key = re.sub(r"\W+", "", adjust_key).lower()
        if normalized_key in normalized_model:
            matches.append((len(normalized_key), adjust_value))

    if matches:
        # Use longest match to prioritize specific variants
        matches.sort(reverse=True, key=lambda x: x[0])
        model_score += matches[0][1]

    final_score = model_score * provider["base_priority"]
    return final_score


def _create_model_config(provider: dict, model_name: str, priority: float) -> dict:
    return {
        "api_key_env_var": provider["api_key_env_var"],
        "base_url": provider["base_url"],
        "model_name": DoubleQuotedScalarString(model_name),
        "priority": priority,
    }


def _write_output_config(config_path: str, output: dict):
    with open(config_path, "w") as f:
        yaml.dump(output, f)


def load_config() -> List[ProviderConfig]:
    loaded_dot_env = load_dotenv(DOT_ENV_FILENAME, verbose=True)
    if not loaded_dot_env:
        logger.warning(f"couldn't load dot env: {DOT_ENV_FILENAME}")
    providers = []

    config_path = os.getenv("CONFIG_PATH", DEFAULT_CONFIG_LOCATION)
    if (
        not os.path.exists(config_path)
        or os.path.getmtime(config_path) < time.time() - GENERATED_PROVIDER_CONFIG_STALE_TIME_SECS
    ):
        generate_providers(config_path)

    with open(config_path, "r") as f:
        config = yaml.load(f)

    for provider in config["providers"]:
        # Get API key from environment variable
        env_var = provider.pop("api_key_env_var")
        api_key = os.getenv(env_var)
        if not api_key:
            raise ValueError(f"Missing environment variable: {env_var}")

        providers.append(
            ProviderConfig(
                base_url=provider["base_url"],
                model_name=provider["model_name"],
                priority=provider["priority"],
                api_key=api_key,
            )
        )

    # remove providers with a priority <1
    providers = [p for p in providers if p.priority >= 1]

    # the higher the priority, the more likely the provider should be used
    providers.sort(key=lambda x: x.priority, reverse=True)
    return providers
