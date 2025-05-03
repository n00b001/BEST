import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Set, Tuple
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
    DEBUG_MODE,
)
from src.model_score import get_leaderboard_score
from src.utils import truncate_dict

logger = logging.getLogger(__name__)
coloredlogs.install(level=LOG_LEVEL, logger=logger)

yaml = ruamel.yaml.YAML()
yaml.preserve_quotes = True
yaml.explicit_start = True
yaml.indent(sequence=4, offset=2)

# Global variable to store allowed authentication tokens - Initialized by load_config
AUTH_TOKENS: Set[str] = set()

class ProviderConfig(BaseModel):
    api_key: str
    base_url: str
    model_name: str
    priority: dict[str, float]

    def __hash__(self):
        # Use a tuple of immutable attributes for hashing
        return hash((
            self.api_key,
            self.base_url,
            self.model_name,
            # Convert priority dict to a frozenset of items for hashing
            frozenset(self.priority.items())
        ))

    def __repr__(self):
        return str(self)

    def __str__(self):
        # Simplified string representation
        return f"ProviderConfig(model='{self.model_name}', base_url='{self.base_url}', score={self.priority.get('overall_score', 'N/A')})"


def _load_model_adjustments(model_adjustments_filename) -> dict:
    try:
        with open(model_adjustments_filename, "r") as f:
            config = yaml.load(f)
        # Basic validation: check if it's a dict and has 'adjustments'
        if isinstance(config, dict) and "adjustments" in config and isinstance(config["adjustments"], dict):
            return config.get("adjustments", {})
        else:
            logger.warning(f"Invalid format in {model_adjustments_filename}. Expected a dictionary with an 'adjustments' key.")
            return {}
    except FileNotFoundError:
        logger.warning(f"No model adjustments found at {model_adjustments_filename}")
        return {}
    except ruamel.yaml.YAMLError as e:
        logger.error(f"Error parsing YAML in {model_adjustments_filename}: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error loading model adjustments from {model_adjustments_filename}: {e}")
        return {}


def generate_providers(providers_config_filename, meta_providers_config_filename, model_adjustments_filename):
    logger.info("Starting provider configuration generation...")
    try:
        with open(meta_providers_config_filename, "r") as f:
            meta_providers_config = yaml.load(f)
            # Validate structure
            if not isinstance(meta_providers_config, dict) or "providers" not in meta_providers_config or not isinstance(meta_providers_config["providers"], list):
                logger.error(f"Invalid format in {meta_providers_config_filename}. Expected dict with a 'providers' list.")
                return
            meta_providers = meta_providers_config["providers"]
    except FileNotFoundError:
        logger.error(f"Meta providers config file not found: {meta_providers_config_filename}")
        return
    except ruamel.yaml.YAMLError as e:
        logger.error(f"Error parsing YAML in {meta_providers_config_filename}: {e}")
        return
    except Exception as e:
        logger.error(f"Error loading meta providers config from {meta_providers_config_filename}: {e}")
        return

    model_adjustments = _load_model_adjustments(model_adjustments_filename)
    output = {"providers": []}
    total_processed_models = 0

    # Process each provider in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=10) as executor: # Limit workers if needed
        # Map each provider to _get_model_data_list
        # This fetches models for all providers concurrently
        model_data_list_of_list = list(executor.map(_get_model_data_list, meta_providers))

        # Prepare arguments for processing models (provider, model_data, adjustments)
        provider_processing_args = [
            (provider, model_data_list, model_adjustments)
            for provider, model_data_list in zip(meta_providers, model_data_list_of_list)
            if model_data_list is not None # Skip providers where fetching failed
        ]

        # Process models for each provider concurrently
        results_list_of_lists = list(executor.map(lambda args: _process_provider_models(*args), provider_processing_args))

        # Collect results
        for provider_models in results_list_of_lists:
            if provider_models:
                output["providers"].extend(provider_models)
                total_processed_models += len(provider_models)

    _write_output_config(providers_config_filename, output)
    logger.info(f"{total_processed_models} provider configurations generated successfully into {providers_config_filename}")


def _get_model_data_list(provider: dict) -> list[dict] | None:
    # Validate provider dictionary structure
    if not isinstance(provider, dict) or "api_key_env_var" not in provider or "base_url" not in provider:
        logger.warning(f"Skipping invalid provider entry: missing required keys. Entry: {provider}")
        return None # Indicate failure

    provider_name = provider.get("base_url", "Unknown Provider") # For logging
    api_key_env_var = provider["api_key_env_var"]
    api_key = os.getenv(api_key_env_var)

    if not api_key:
        logger.warning(f"Skipping {provider_name} - API key environment variable '{api_key_env_var}' not set")
        return None # Indicate failure

    try:
        model_data_list = _fetch_provider_models(provider, api_key)
        if model_data_list is None: # Check if fetching failed
             return None
        logger.debug(f"Successfully fetched {len(model_data_list)} models for {provider_name}")
        return model_data_list
    except Exception as e: # Catch any unexpected error during fetching/parsing
        logger.error(f"Error fetching or processing models for {provider_name}: {e}", exc_info=True)
        return None # Indicate failure


def _fetch_provider_models(provider: dict, api_key: str) -> list | None:
    model_url = provider.get("model_url", provider["base_url"]) # Fallback to base_url
    provider_name = provider.get("base_url", "Unknown Provider") # For logging

    # Construct the full URL for the models endpoint
    # Handle base URLs that might or might not end with a slash
    models_endpoint_url = f"{model_url.rstrip('/')}/models"

    try:
        response = requests.get(
            models_endpoint_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=15 # Increased timeout slightly
        )
        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
        models_data = response.json()

        # Handle different expected response structures
        if isinstance(models_data, dict):
            model_list = models_data.get("data", models_data.get("models")) # Check common keys
            if isinstance(model_list, list):
                return model_list
            else:
                logger.warning(f"Unexpected model list format from {provider_name} ({models_endpoint_url}). Expected list under 'data' or 'models', got: {type(model_list)}")
                return [] # Return empty list on format error
        elif isinstance(models_data, list):
            return models_data # Response is directly a list of models
        else:
            logger.warning(f"Unexpected response format from {provider_name} ({models_endpoint_url}). Expected dict or list, got: {type(models_data)}")
            return [] # Return empty list on format error

    except requests.exceptions.Timeout:
        logger.error(f"Timeout fetching models from {provider_name} ({models_endpoint_url})")
        return None # Indicate failure distinctly from empty list
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error fetching models from {provider_name} ({models_endpoint_url}): {e}")
        # Log response body if possible and helpful (be careful with large responses)
        if e.response is not None:
             logger.error(f"Response body: {e.response.text[:500]}...") # Log first 500 chars
        return None # Indicate failure
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching models from {provider_name} ({models_endpoint_url}): {e}")
        return None # Indicate failure
    except Exception as e:
        logger.error(f"An unexpected error occurred while fetching models from {provider_name} ({models_endpoint_url}): {e}", exc_info=True)
        return None # Indicate failure


def _process_provider_models(provider: dict, model_data_list: list, model_adjustments: dict) -> list[dict]:
    processed_models = []
    provider_base_url = provider.get('base_url', 'Unknown Provider')
    provider_netloc = urlparse(provider_base_url).netloc if provider_base_url != 'Unknown Provider' else 'Unknown Provider'

    if not model_data_list:
         # This case might be handled by the caller checking for None, but good to have defense here too
         logger.debug(f"No model data provided for processing: {provider_netloc}")
         return []

    processed_count = 0
    skipped_count = 0
    for model in model_data_list:
        if not isinstance(model, dict):
             logger.warning(f"Skipping invalid model entry (not a dict) for {provider_netloc}: {model}")
             skipped_count += 1
             continue

        model_name = model.get("id", model.get("name")) # Prefer 'id', fallback to 'name'
        if not model_name:
            logger.warning(f"Skipping model for {provider_netloc} due to missing 'id' or 'name': {truncate_dict(model)}")
            skipped_count += 1
            continue

        # Gracefully handle missing or invalid context_length
        context_length = model.get("context_length", model.get("max_position_embeddings")) # Check common keys
        if not isinstance(context_length, (int, float)) or context_length < 0:
            logger.debug(f"Invalid or missing context_length for model {model_name} from {provider_netloc}. Defaulting to 0. Value: '{context_length}'")
            context_length = 0
        else:
            context_length = int(context_length) # Ensure integer

        try:
            priority = _calculate_model_priority(provider, model_name, context_length, model_adjustments)
            processed_models.append(_create_model_config(provider, model_name, priority))
            processed_count += 1
        except Exception as e:
            logger.error(f"Error calculating priority or creating config for model {model_name} from {provider_netloc}: {e}", exc_info=True)
            skipped_count += 1

    if skipped_count > 0:
        logger.warning(f"Skipped {skipped_count} model entries for {provider_netloc} during processing.")
    logger.info(f"Successfully processed {processed_count} models for: {provider_netloc}")
    return processed_models


def _calculate_model_priority(
    provider: dict, model_name: str, context_length: int, model_adjustments: dict
) -> dict[str, float]:

    # 1. Leaderboard Score
    leaderboard_score = get_leaderboard_score(model_name) * MODEL_LEADERBOARD_SCORE_SCALAR

    # 2. Parameter-based Scoring (Handles b, m, k suffixes)
    model_param_score = 0.0
    # Simple model size (e.g., -7b, -13m, -80k)
    if match := re.search(r"-(\d+\.?\d*)[bBmMkK]", model_name):
        param_num = float(match.group(1))
        suffix = match.group(0).lower()[-1]
        if suffix == 'm':
            param_num *= 1e6
        elif suffix == 'b':
            param_num *= 1e9
        elif suffix == 'k':
             param_num *= 1e3 # Assuming k = thousands
        model_param_score = param_num * MODEL_PARAM_SCORE_SCALAR
    # Mixtral-style (e.g., 8x7b)
    elif x_match := re.search(r"(\d+)x(\d+\.?\d*)[bBmMkK]", model_name):
        experts = float(x_match.group(1))
        param_per_expert = float(x_match.group(2))
        suffix = x_match.group(0).lower()[-1]
        if suffix == 'm':
            param_per_expert *= 1e6
        elif suffix == 'b':
            param_per_expert *= 1e9
        elif suffix == 'k':
             param_per_expert *= 1e3
        model_param_score = experts * param_per_expert * MODEL_PARAM_SCORE_SCALAR

    # 3. Context Length Scoring
    model_context_score = context_length * MODEL_CTX_SCORE_SCALAR

    # 4. Model Adjustment Score (from config/model-adjustments.yaml)
    model_adjustment_score = 0.0
    applied_adjustments = [] # Keep track of adjustments applied
    for group_name, group_config in model_adjustments.items():
        if not isinstance(group_config, dict) or "priority" not in group_config or "regex" not in group_config:
            logger.warning(f"Skipping invalid adjustment group '{group_name}': missing 'priority' or 'regex'")
            continue

        group_priority = group_config.get("priority", 0.0)
        if not isinstance(group_priority, (int, float)):
             logger.warning(f"Skipping invalid priority value for group '{group_name}': {group_priority}")
             continue

        regex_patterns = group_config["regex"]
        if not isinstance(regex_patterns, list):
            logger.warning(f"Skipping invalid regex patterns for group '{group_name}': not a list.")
            continue

        for pattern_str in regex_patterns:
            if not isinstance(pattern_str, str):
                logger.warning(f"Skipping non-string regex pattern in group '{group_name}': {pattern_str}")
                continue
            try:
                # Compile with IGNORECASE and match anywhere in the string
                pattern = re.compile(pattern_str, re.IGNORECASE)
                if pattern.search(model_name):
                    logger.debug(f"Regex '{pattern_str}' matched model '{model_name}' (group: '{group_name}'), adding priority: {group_priority}")
                    model_adjustment_score += group_priority
                    applied_adjustments.append(f"{group_name}({group_priority})") # Record adjustment
                    break # Apply only the first matching pattern within a group
            except re.error as e:
                logger.error(f"Invalid regex pattern '{pattern_str}' in group '{group_name}': {e}")
                continue # Skip this pattern

    model_adjustment_score *= MODEL_ADJUSTMENT_SCALAR

    # 5. Provider Base Priority
    provider_base_priority = provider.get("base_priority", 1.0)
    if not isinstance(provider_base_priority, (int, float)):
        logger.warning(f"Invalid base_priority for provider '{provider.get("base_url")}': {provider_base_priority}. Using 1.0.")
        provider_base_priority = 1.0

    # 6. Calculate Overall Score
    overall_score = (
        leaderboard_score
        + model_param_score
        + model_context_score
        + model_adjustment_score
    ) * provider_base_priority

    # Create the priority dictionary
    score_dict = {
        "base_priority": provider_base_priority,
        "leaderboard_score": round(leaderboard_score, 4),
        "model_param_score": round(model_param_score, 4),
        "model_context_score": round(model_context_score, 4),
        "model_adjustment_score": round(model_adjustment_score, 4),
        "applied_adjustments": ", ".join(applied_adjustments) if applied_adjustments else "None", # Add applied adjustments info
        "overall_score": round(overall_score, 4), # Round final score
    }

    logger.debug(f"Calculated priority for '{model_name}': {score_dict}")
    return score_dict


def _create_model_config(provider: dict, model_name: str, priority: dict[str, float]) -> dict:
    # Ensure model_name is quoted for YAML safety, especially if it contains special chars
    quoted_model_name = DoubleQuotedScalarString(model_name)
    return {
        "api_key_env_var": provider["api_key_env_var"],
        "base_url": provider["base_url"],
        "model_name": quoted_model_name,
        "priority": priority, # Store the detailed priority dict
    }


def _write_output_config(config_path: str, output: dict):
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            yaml.dump(output, f)
        logger.debug(f"Successfully wrote provider config to {config_path}")
    except IOError as e:
        logger.error(f"Error writing provider config to {config_path}: {e}")
    except ruamel.yaml.YAMLError as e:
        logger.error(f"Error formatting YAML for {config_path}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error writing config to {config_path}: {e}")


def load_config() -> Tuple[List[ProviderConfig], Set[str]]:
    global AUTH_TOKENS # Declare intent to modify global variable

    # Load .env file
    loaded_dot_env = load_dotenv(DOT_ENV_FILENAME, override=True, verbose=True) # Override existing env vars if present in .env
    if loaded_dot_env:
        logger.info(f"Loaded environment variables from: {DOT_ENV_FILENAME}")
    else:
        logger.warning(f"Could not load .env file from: {DOT_ENV_FILENAME}. Relying on existing environment variables.")

    # Load authentication tokens from AUTH_TOKENS environment variable
    auth_tokens_str = os.getenv("AUTH_TOKENS", "")
    if auth_tokens_str:
        # Split by comma, strip whitespace, filter out empty strings
        current_auth_tokens = set(token.strip() for token in auth_tokens_str.split(',') if token.strip())
        if len(current_auth_tokens) != len(auth_tokens_str.split(',')):
             logger.warning("Some empty or whitespace-only tokens were found in AUTH_TOKENS and ignored.")
        AUTH_TOKENS = current_auth_tokens
        logger.info(f"Loaded {len(AUTH_TOKENS)} authentication tokens.")
        # Avoid logging tokens themselves unless in extreme debug situations
        # logger.debug(f"Loaded tokens: {AUTH_TOKENS}")
    else:
        AUTH_TOKENS = set() # Ensure it's an empty set if env var is missing or empty
        logger.warning("AUTH_TOKENS environment variable not set or empty. API authentication will be DISABLED.")

    # Determine config file paths from environment or use defaults
    # Use os.path.abspath to ensure paths are absolute, helping avoid CWD issues
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Assumes config.py is in src/
    providers_config_filename = os.getenv("PROVIDERS_CONFIG_FILENAME") or os.path.join(base_dir, PROVIDERS_CONFIG_FILENAME)
    meta_providers_config_filename = os.getenv("META_PROVIDERS_CONFIG_FILENAME") or os.path.join(base_dir, META_PROVIDERS_CONFIG_FILENAME)
    model_adjustments_filename = os.getenv("MODEL_ADJUSTMENTS_FILENAME") or os.path.join(base_dir, MODEL_ADJUSTMENTS_FILENAME)

    logger.info(f"Using Providers Config: {providers_config_filename}")
    logger.info(f"Using Meta Providers Config: {meta_providers_config_filename}")
    logger.info(f"Using Model Adjustments: {model_adjustments_filename}")

    # Generate providers config if needed
    should_generate = False
    if not os.path.exists(providers_config_filename):
        logger.info(f"Generating providers config: file not found at {providers_config_filename}")
        should_generate = True
    else:
        try:
            file_mod_time = os.path.getmtime(providers_config_filename)
            stale_threshold = time.time() - GENERATED_PROVIDER_CONFIG_STALE_TIME_SECS
            if file_mod_time < stale_threshold:
                logger.info(f"Generating providers config: file is stale (older than {GENERATED_PROVIDER_CONFIG_STALE_TIME_SECS} seconds).")
                should_generate = True
        except OSError as e:
            logger.warning(f"Could not check modification time for {providers_config_filename}: {e}. Will not regenerate based on staleness.")

    if DEBUG_MODE and not should_generate:
        logger.info("Generating providers config: DEBUG_MODE is enabled.")
        should_generate = True

    if should_generate:
        # Check if meta-providers file exists before attempting generation
        if os.path.exists(meta_providers_config_filename):
            generate_providers(providers_config_filename, meta_providers_config_filename, model_adjustments_filename)
        else:
            logger.error(f"Cannot generate providers config because meta providers file is missing: {meta_providers_config_filename}")

    # Load the generated or existing providers config
    providers_config_data = None
    if os.path.exists(providers_config_filename):
        try:
            with open(providers_config_filename, "r") as f:
                providers_config_data = yaml.load(f)
            if not isinstance(providers_config_data, dict) or "providers" not in providers_config_data or not isinstance(providers_config_data["providers"], list):
                logger.error(f"Invalid format in provider config file: {providers_config_filename}. Expected dict with 'providers' list.")
                providers_config_data = {"providers": []}
        except ruamel.yaml.YAMLError as e:
            logger.error(f"Error parsing YAML in {providers_config_filename}: {e}")
            providers_config_data = {"providers": []}
        except Exception as e:
            logger.error(f"Error loading provider config file {providers_config_filename}: {e}")
            providers_config_data = {"providers": []}
    else:
        logger.error(f"Provider config file not found: {providers_config_filename}. No providers will be loaded.")
        providers_config_data = {"providers": []}

    # Process provider configurations into ProviderConfig objects
    providers: List[ProviderConfig] = []
    provider_entries = providers_config_data.get("providers", [])
    successfully_loaded_count = 0
    failed_load_count = 0

    for provider_data in provider_entries:
        if not isinstance(provider_data, dict):
            logger.warning(f"Skipping invalid provider entry (not a dict): {provider_data}")
            failed_load_count += 1
            continue

        required_keys = ["api_key_env_var", "base_url", "model_name", "priority"]
        missing_keys = [key for key in required_keys if key not in provider_data]
        if missing_keys:
            logger.warning(f"Skipping provider due to missing keys: {missing_keys}. Data: {truncate_dict(provider_data)}")
            failed_load_count += 1
            continue

        env_var = provider_data["api_key_env_var"]
        api_key = os.getenv(env_var)
        if not api_key:
            logger.warning(f"API key env var '{env_var}' not set for model '{provider_data.get('model_name')}' at '{provider_data.get('base_url')}'. Skipping provider.")
            failed_load_count += 1
            continue

        priority_dict = provider_data["priority"]
        if not isinstance(priority_dict, dict) or "overall_score" not in priority_dict:
            logger.warning(f"Skipping provider '{provider_data.get('model_name')}' due to invalid or missing 'overall_score' in priority. Data: {priority_dict}")
            failed_load_count += 1
            continue

        try:
            # Basic type validation before creating Pydantic model
            if not isinstance(provider_data["base_url"], str) or not isinstance(provider_data["model_name"], str):
                 raise ValueError("base_url and model_name must be strings.")

            provider_obj = ProviderConfig(
                base_url=provider_data["base_url"],
                model_name=provider_data["model_name"],
                priority=priority_dict,
                api_key=api_key,
            )
            # Filter out providers with score <= 0 AFTER creation
            if provider_obj.priority.get("overall_score", 0) > 0:
                 providers.append(provider_obj)
                 successfully_loaded_count += 1
            else:
                 logger.debug(f"Filtered out low-scoring provider: {provider_obj}")
                 failed_load_count += 1 # Count low-scoring as 'failed' to load into active set

        except (ValueError, TypeError) as e: # Catch Pydantic validation errors etc.
            logger.error(f"Error creating ProviderConfig for '{provider_data.get('model_name', 'Unknown Model')}': {e}. Data: {truncate_dict(provider_data)}")
            failed_load_count += 1
        except Exception as e:
            logger.error(f"Unexpected error processing provider '{provider_data.get('model_name', 'Unknown Model')}': {e}", exc_info=True)
            failed_load_count += 1

    if failed_load_count > 0:
         logger.warning(f"Skipped or failed to load {failed_load_count} provider configurations.")

    # Sort the valid, high-scoring providers
    sort_providers(providers)

    logger.info(f"Loaded and sorted {len(providers)} active providers (overall_score > 0).")
    return providers, AUTH_TOKENS


def sort_providers(providers: List[ProviderConfig]):
    # Sorts the list in-place: higher overall_score first
    providers.sort(key=lambda x: x.priority.get("overall_score", 0), reverse=True)
    logger.debug(f"Sorted {len(providers)} providers.")


# Note: A dedicated function/mechanism would be needed for true config reloading
# without restarting the server process (e.g., using signals or a watcher thread).
# The current lifespan approach reloads config on startup/restart.
