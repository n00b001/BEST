# B.E.S.T. – Benchmark‑Enhanced Selective Transit

A unified API gateway for multiple OpenAI-compatible LLM providers. This project allows you to use various LLM services
through a single, OpenAI-compatible API, simplifying integration and providing features like failover and rate limiting.

## Authentication

Access to the LLM Gateway API (excluding health checks and OpenAPI docs) is protected by API key authentication. You must include an API key with your requests.

API keys are passed using the `Authorization` header with the `Bearer` scheme.

**Example using `curl`:**

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
     -H "Authorization: Bearer llmgw-sk-yourgeneratedapikey" \
     -H "Content-Type: application/json" \
     -d '{
           "model": "priority-auto",
           "messages": [{"role": "user", "content": "Hello!"}]
         }'
```

**Example using the `openai` Python client:**

Ensure your OpenAI client is configured to point to the LLM Gateway and use your API key:

```python
import openai

# Configure the base URL to point to your LLM Gateway instance
openai.api_base = "http://localhost:8000/v1" # Or your deployed gateway URL

# Set your API key obtained from the LLM Gateway admin interface
openai.api_key = "llmgw-sk-yourgeneratedapikey" 

# Make a request
try:
    chat_completion = openai.ChatCompletion.create(
        model="priority-auto", # Or a specific model available through the gateway
        messages=[{"role": "user", "content": "Translate 'hello' to French."}]
    )
    print(chat_completion.choices[0].message.content)
except openai.error.OpenAIError as e:
    print(f"An API error occurred: {e}")

```

API keys must be generated using the Admin API. See the "API Key Management (Admin)" section for details.

## API Key Management (Admin)

API keys for accessing the LLM Gateway are managed through a set of administrative endpoints. These endpoints are protected by a separate Admin API Key.

**Admin Authentication:**

*   The Admin API Key must be set using the `LLMGW_ADMIN_API_KEY` environment variable when running the gateway (e.g., in your `.env` file).
*   Requests to admin endpoints must include this key in the `X-Admin-API-Key` header.

**Admin Endpoints:**

*   **`POST /admin/keys`**: Generate a new API key.
    *   **Request Body**: `{"name": "your_key_name"}`
    *   **Response**: Details of the new key, including the `plaintext_key`. **Store this key securely as it will not be shown again.**
*   **`GET /admin/keys`**: List all active (non-revoked) API keys. (Does not show the key itself).
*   **`DELETE /admin/keys/{key_id}`**: Revoke an API key by its ID.
*   **`GET /admin/usage`**: (Placeholder) View basic usage statistics by key (currently shows key details including `last_used`).

**Example: Creating an API Key using `curl`**

```bash
curl -X POST http://localhost:8000/admin/keys \
     -H "X-Admin-API-Key: your_secret_admin_key" \
     -H "Content-Type: application/json" \
     -d '{"name": "my_test_key"}'
```

**Example: Listing API Keys using `curl`**

```bash
curl -X GET http://localhost:8000/admin/keys \
     -H "X-Admin-API-Key: your_secret_admin_key"
```

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
