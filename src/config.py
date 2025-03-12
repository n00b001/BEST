import logging
import os
import re
from datetime import datetime
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
)

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
    priority: int
    cooldown_until: datetime


def generate_providers(config_path):
    # Load meta providers configuration
    with open(META_PROVIDERS_CONFIG_FILENAME, "r") as f:
        meta_providers = yaml.load(f)["providers"]

    output = {"providers": []}

    for provider in meta_providers:
        # Get API key from environment variable
        api_key = os.getenv(provider["api_key_env_var"])
        if not api_key:
            logger.warning(f"Skipping provider {provider['base_url']} - API key not found in environment")
            continue

        # Fetch models from provider
        try:
            response = requests.get(
                f"{provider['base_url']}/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            response.raise_for_status()
            models_data = response.json()
        except Exception as e:
            logger.error(f"Error fetching models from {provider['base_url']}: {e}")
            continue

        models = []
        # Process each model
        for model in models_data.get("data", []):
            model_name = model.get("id", "")

            # todo get priority based on score from SWEbench (or similar).
            #  If model can't be found in benchmark, then fallback to priority based on parameters

            # Calculate model priority
            model_priority = 1
            if match := re.search(r"-(\d+)b", model_name):
                model_priority += int(match.group(1))
            # todo add the ability to override priority for models (with some config)
            #  either very well known good models (priority goes up)
            #  or bad models that are known to be terrible (priority goes down)

            final_priority = model_priority * provider["base_priority"]

            models.append(
                {
                    "api_key_env_var": provider["api_key_env_var"],
                    "base_url": provider["base_url"],
                    "model_name": DoubleQuotedScalarString(model_name),
                    "priority": final_priority,
                }
            )
        logger.info(f"Got {len(models)} models for: {urlparse(provider['base_url']).netloc}")
        output["providers"].extend(models)

    # Write output to providers.yaml
    with open(config_path, "w") as f:
        # yaml.dump(output, f, Dumper=MyDumper, sort_keys=False, default_flow_style=False)
        yaml.dump(output, f)

    logger.info("Providers configuration generated successfully")


def load_config() -> List[ProviderConfig]:
    loaded_dot_env = load_dotenv(DOT_ENV_FILENAME, verbose=True)
    if not loaded_dot_env:
        raise RuntimeError(f"couldn't load dot env: {DOT_ENV_FILENAME}")
    providers = []

    config_path = os.getenv("CONFIG_PATH", DEFAULT_CONFIG_LOCATION)
    # generate providers.yaml
    generate_providers(config_path)

    with open(config_path, "r") as f:
        config = yaml.load(f)

    for provider in config["providers"]:
        # Get API key from environment variable
        env_var = provider.pop("api_key_env_var")
        api_key = os.getenv(env_var)
        if not api_key:
            raise ValueError(f"Missing environment variable: {env_var}")

        providers.append(ProviderConfig(**provider, api_key=api_key))

    # todo: remove providers with a priority <1

    # the higher the priority, the more likely the provider should be used
    providers.sort(key=lambda x: x.priority, reverse=True)
    return providers
