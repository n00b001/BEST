import os

import os
from typing import Literal

PORT: int = 12345
API_TIMEOUT_SECS: int = 300
LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
DOT_ENV_FILENAME: str = ".env"
DEFAULT_CONFIG_FOLDER: str = "config"
DEFAULT_CONFIG_LOCATION: str = os.path.join(DEFAULT_CONFIG_FOLDER, "providers.yaml")
META_PROVIDERS_CONFIG_FILENAME: str = os.path.join(DEFAULT_CONFIG_FOLDER, "meta-providers.yaml")
MODEL_ADJUSTMENTS_FILENAME: str = os.path.join(DEFAULT_CONFIG_FOLDER, "model-adjustments.yaml")
DEFAULT_COOLDOWN_SECONDS: int = 30
BAD_REQUEST_COOLDOWN_SECONDS: int = 300
HEALTHCHECK_URL: str = "https://example.com"
MAX_REQUEST_CHAR_COUNT_FOR_LOG: int = 100
GENERATED_PROVIDER_CONFIG_STALE_TIME_SECS: int = 60 * 60 * 24

# model params are typically in the range: 0.5b -> 405b
MODEL_PARAM_SCORE_SCALAR = 0.1
# context is typicall in the range: 1024 -> 1000000
MODEL_CTX_SCORE_SCALAR = 0.01
