import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
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
    PROVIDERS_CONFIG_FILENAME,
    META_PROVIDERS_CONFIG_FILENAME,
    GENERATED_PROVIDER_CONFIG_STALE_TIME_SECS,
    MODEL_PARAM_SCORE_SCALAR,
    MODEL_CTX_SCORE_SCALAR,
    MODEL_ADJUSTMENTS_FILENAME,
    MODEL_LEADERBOARD_SCORE_SCALAR,
    MODEL_ADJUSTMENT_SCALAR,
)
from src.model_score import get_leaderboard_score
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
    priority: dict[str, float]

    def __hash__(self):
        return hash(f"{self.api_key}{self.base_url}{self.model_name}{str(property)}")
    def __repr__(self):
        return str(self)
    def __str__(self):
        return f"{self.model_name}"


def _load_model_adjustments(model_adjustments_filename) -> dict:
    try:
        with open(model_adjustments_filename, "r") as f:
            config = yaml.load(f)
        return config.get("adjustments", {})
    except FileNotFoundError:
        logger.warning(f"No model adjustments found at {model_adjustments_filename}")
        return {}
    except Exception as e:
        logger.error(f"Error loading model adjustments: {e}")
        return {}


def generate_providers(providers_config_filename, meta_providers_config_filename, model_adjustments_filename):
    with open(meta_providers_config_filename, "r") as f:
        meta_providers = yaml.load(f)["providers"]

    model_adjustments = _load_model_adjustments(model_adjustments_filename)
    output = {"providers": []}

    # Process each provider in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        # Prepare arguments for each provider
        model_adjustments_list = [model_adjustments] * len(meta_providers)
        # Map each provider to _process_provider with model_adjustments
        model_data_list_of_list = executor.map(_get_model_data_list, meta_providers)

        for provider, model_data_list, model_adjustments in zip(
            meta_providers, model_data_list_of_list, model_adjustments_list
        ):
            results = _process_provider_models(provider, model_data_list, model_adjustments)

            # Collect results in the order of meta_providers
            for provider_models in results:
                if provider_models:
                    output["providers"].append(provider_models)

    _write_output_config(providers_config_filename, output)
    logger.info(f"{len(output['providers'])} providers configuration generated successfully")


def _get_model_data_list(provider: dict) -> list[dict]:
    api_key = os.getenv(provider["api_key_env_var"])
    if not api_key:
        logger.warning(f"Skipping {provider['base_url']} - API key missing")
        return []

    try:
        model_data_list = _fetch_provider_models(provider, api_key)
        return model_data_list
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


def _calculate_model_priority(
    provider: dict, model_name: str, context_length: int, model_adjustments: dict
) -> dict[str, float]:
    leaderboard_score = get_leaderboard_score(model_name) * MODEL_LEADERBOARD_SCORE_SCALAR

    # Parameter-based scoring
    model_param_score = 0.0
    if match := re.search(r"-(\d+)[bBmM]", model_name):
        model_param_score += float(match.group(1)) * MODEL_PARAM_SCORE_SCALAR

    # Mixtral-style model scoring
    if x_match := re.search(r"(\d+)x(\d+)[bBmM]", model_name):
        model_param_score += float(x_match.group(1)) * int(x_match.group(2)) * MODEL_PARAM_SCORE_SCALAR

    # Context length scoring
    model_context_score = context_length * MODEL_CTX_SCORE_SCALAR

    # Apply global model adjustments with regex matching
    model_adjustment_score = 0.0
    for group_name, group_config in model_adjustments.items():
        group_priority = group_config["priority"]
        regex_patterns = group_config["regex"]

        for pattern_str in regex_patterns:
            try:
                pattern = re.compile(pattern_str, re.IGNORECASE)
                if pattern.search(model_name):
                    logger.info(f"{pattern_str} matched: {model_name}, adding: {group_priority}")
                    model_adjustment_score += group_priority
                    break  # Only apply once per model group if any pattern matches
            except re.error as e:
                logger.error(f"Invalid regex pattern '{pattern_str}': {e}")
                continue

    model_adjustment_score *= MODEL_ADJUSTMENT_SCALAR

    overall_score = (leaderboard_score + model_param_score + model_context_score + model_adjustment_score) * provider[
        "base_priority"
    ]
    score_dict = {
        "base": provider["base_priority"],
        "leaderboard_score": leaderboard_score,
        "model_param_score": model_param_score,
        "model_context_score": model_context_score,
        "model_adjustment_score": model_adjustment_score,
        "overall_score": overall_score,
    }
    return score_dict


def _create_model_config(provider: dict, model_name: str, priority: dict[str, float]) -> dict:
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

    providers_config_filename = os.getenv("PROVIDERS_CONFIG_FILENAME", PROVIDERS_CONFIG_FILENAME)
    meta_providers_config_filename = os.getenv("META_PROVIDERS_CONFIG_FILENAME", META_PROVIDERS_CONFIG_FILENAME)
    model_adjustments_filename = os.getenv("MODEL_ADJUSTMENTS_FILENAME", MODEL_ADJUSTMENTS_FILENAME)

    if (
        not os.path.exists(providers_config_filename)
        or os.path.getmtime(providers_config_filename) < time.time() - GENERATED_PROVIDER_CONFIG_STALE_TIME_SECS
    ):
        generate_providers(providers_config_filename, meta_providers_config_filename, model_adjustments_filename)

    with open(providers_config_filename, "r") as f:
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

    # the higher the priority, the more likely the provider should be used
    providers.sort(key=lambda x: x.priority["overall_score"], reverse=True)

    # remove providers with a priority <1
    high_scoring_providers = set([p for p in providers if p.priority["overall_score"] > 0])
    low_scoring_providers = set(providers).difference(high_scoring_providers)

    if len(low_scoring_providers) > 0:
        logger.warning(f"Removed {len(low_scoring_providers)} providers where score < 1")
        for p in low_scoring_providers:
            logger.debug(p)

    return providers
