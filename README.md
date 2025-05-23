# B.E.S.T. – Benchmark‑Enhanced Selective Transit

A unified API gateway for multiple OpenAI-compatible LLM providers. This project allows you to use various LLM services
through a single, OpenAI-compatible API, simplifying integration and providing features like failover and rate limiting.

## Getting Started

1. **Install Dependencies:** Install the required packages using:
    - `uv sync` (https://docs.astral.sh/uv/)

2. **Configure Providers:**
    - Add your LLM providers to `config/meta-providers.yaml`.
    - Set up your API keys in a `.env` file. You can find resources for API keys
      here: [https://github.com/cheahjs/free-llm-api-resources](https://github.com/cheahjs/free-llm-api-resources)

3. **Start the Service:**
    - Run the gateway using: `uv run python -m uvicorn src.main:app --port 12345`

4. **Use the API:**
    - Point your OpenAI client to the gateway:
    ```python
    import openai

    openai.api_base = "http://localhost:8000/v1"
    openai.api_key = "any-key"  # Not used by the gateway
    ```

## Dynamic Scoring Mechanism

The LLM Gateway uses a dynamic scoring mechanism to prioritize and route requests to the most suitable provider. The score for each provider is calculated based on the following factors:

1. **Priority**: A static priority value assigned to each provider for a specific model.
2. **Leaderboard Score**: A score fetched from a leaderboard (e.g., BigCodeBench) that reflects the model's performance.
3. **Parameter Score**: A score calculated based on the model's parameters (e.g., `max_tokens`, `temperature`).
4. **Context Score**: A score calculated based on the context length of the model.

The final score is a weighted sum of these factors, where the weights are defined by the constants `MODEL_LEADERBOARD_SCORE_SCALAR`, `MODEL_PARAM_SCORE_SCALAR`, and `MODEL_CTX_SCORE_SCALAR`.

## Features

- Single API endpoint matching OpenAI's specification.
- Automatic failover to backup providers.
- Rate limit handling and cooldown management.
- Priority-based routing.
- Async requests for improved performance.
- Health check endpoint.
- Robust error handling.

## Authentication

The gateway supports Bearer Token authentication for securing its endpoints. This allows you to control who can access the API.

### Configuration

Authentication is configured using the `ALLOWED_BEARER_TOKENS` environment variable. This variable should contain a comma-separated list of valid bearer tokens.

**Example:**

```bash
export ALLOWED_BEARER_TOKENS="your_secret_token1,another_secure_token,token3"
```

Or, you can add this line to your `.env` file:

```
ALLOWED_BEARER_TOKENS="your_secret_token1,another_secure_token,token3"
```

### Behavior

-   **Authentication Enabled**: If `ALLOWED_BEARER_TOKENS` is set and contains one or more tokens, clients must provide a valid token in the `Authorization` header with the "Bearer" scheme.
    Example: `Authorization: Bearer your_secret_token1`
-   **Authentication Disabled**: If the `ALLOWED_BEARER_TOKENS` environment variable is not set or is left empty, authentication is disabled. In this mode, all protected routes will be accessible without requiring a token. This is useful for development or trusted environments.

The following endpoints are protected by authentication when it is enabled:
- `/chat/completions`
- `/v1/chat/completions`
- `/stats`
- `/models`
- `/v1/models`

The `/health` and `/ok` endpoints are always accessible without authentication.

## **Project Rules**

- Always update requirements.txt when adding dependencies
- Keep the README updated with current status
- Write tests for new features
- Maintain API compatibility with OpenAI's spec
- Document all configuration options
- After any change, run the project to make sure it still works
- After any change, run any tests to make sure they still pass

## This implementation provides:

1. OpenAI-compatible API endpoint
2. Priority-based routing with failover
3. Rate limit handling with Retry-After parsing
4. Async requests for better performance
5. Clean configuration management
6. Health check endpoint
7. Proper error handling

## To use it:

1. Configure your LLM providers in `config/meta-providers.yaml`
2. Setup up your API keys in `.env`
3. Start the service with `uv run python -m uvicorn src.main:app --port 12345`
4. Use it like you would the OpenAI API:

```python
import openai

openai.api_base = "http://localhost:8000/v1"
openai.api_key = "any-key"  # Not used by gateway
```

## To develop:

1. `uv sync --dev`
2. `uv run python -m uvicorn src.main:app --reload --port 12345`
3. And after making code changes, run this before pushing (if it fails here, it will fail github actions) `uv run python -m autopep8 --exclude .venv -ri . && uv run python -m black --fast --color -l 120 . && uv run python -m mypy --exclude .venv --follow-untyped-imports --explicit-package-bases . && uv run python -m flake8 --exclude .venv --max-line-length 120 . && uv run python -m pylint --ignore .venv --output-format=colorized \
          --max-line-length 120 --fail-under 5 --fail-on E . && uv run python -m pytest --color yes --verbosity=3;`

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=n00b001/BEST&type=Date)](https://www.star-history.com/#n00b001/BEST&Date)

## Todo

- Add more concrete examples to the documentation.
