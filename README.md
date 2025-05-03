# B.E.S.T. – Benchmark‑Enhanced Selective Transit

A unified API gateway for multiple OpenAI-compatible LLM providers. This project allows you to use various LLM services
through a single, OpenAI-compatible API, simplifying integration and providing features like failover, dynamic scoring, and **authentication**.

## Getting Started

1.  **Install Dependencies:** Install the required packages using:
    *   `uv sync` (using [uv](https://docs.astral.sh/uv/))

2.  **Configure Providers:**
    *   Add your LLM providers to `config/meta-providers.yaml`.
    *   Set up your API keys for *each provider* in a `.env` file (copy `.env.example` if needed). You can find resources for API keys here: [https://github.com/cheahjs/free-llm-api-resources](https://github.com/cheahjs/free-llm-api-resources)

3.  **Configure Authentication (Required):**
    *   Define a comma-separated list of valid **Bearer tokens** in your `.env` file using the `AUTH_TOKENS` variable. These are the tokens clients will need to present to access the gateway.
    *   Example `.env` entry:
        ```dotenv
        # .env
        PROVIDER_ONE_API_KEY=sk-yourproviderkey1...
        PROVIDER_TWO_API_KEY=yourproviderkey2...
        # ... other provider keys ...

        # Comma-separated list of tokens allowed to access the gateway
        AUTH_TOKENS=my-secret-token-1,another-valid-token,user-token-abcdef
        ```
    *   **Security Note:** Choose strong, unpredictable tokens. If `AUTH_TOKENS` is not set or is empty, authentication will be **disabled**, allowing unauthenticated access (not recommended for production).

4.  **Start the Service:**
    *   Run the gateway using: `uv run python src/main.py` (The port is configured via the `PORT` env var, default is 12345)
    *   OR for development with auto-reload: `uv run python -m uvicorn src.main:app --reload --port 12345`

5.  **Use the API:**
    *   Point your OpenAI client (or compatible tool like curl) to the gateway, providing one of the configured `AUTH_TOKENS` as the Bearer token in the `Authorization` header.
    *   **Python Client Example:**
        ```python
        import openai
        import os # For loading environment variables

        # Load your gateway token from environment variable or directly
        gateway_token = os.getenv("MY_GATEWAY_TOKEN", "my-secret-token-1") # Use a token from your AUTH_TOKENS list

        # The base URL of your running LLM Gateway instance (use HTTPS if applicable)
        # Ensure port matches your setup (default 12345 or from PORT env var)
        gateway_base_url = os.getenv("LLM_GATEWAY_URL", "http://localhost:12345")

        client = openai.OpenAI(
            api_key=gateway_token,
            base_url=f"{gateway_base_url}/v1"
        )

        try:
            response = client.chat.completions.create(
                model="priority-auto", # Or specify a model like "gpt-4", "claude-3-opus-20240229"
                messages=[{"role": "user", "content": "Hello! What models are you connected to?"}]
            )
            print(response.choices[0].message.content)
        except openai.AuthenticationError as e:
            print(f"Authentication Error: {e}. Ensure your gateway token is correct and present in the gateway's AUTH_TOKENS.")
            # You might want to check the status_code and body attributes of the error object
            # print(f"Status Code: {e.status_code}")
            # print(f"Response Body: {e.body}")
        except openai.APIConnectionError as e:
            print(f"Connection Error: {e}. Is the gateway running at {gateway_base_url}?")
        except openai.APIError as e:
            print(f"API Error: {e}")

        ```
    *   **Curl Example:**
        ```bash
        # Make sure to replace YOUR_GATEWAY_TOKEN with a valid token from your AUTH_TOKENS list
        export YOUR_GATEWAY_TOKEN="my-secret-token-1"

        curl http://localhost:12345/v1/chat/completions \
          -H "Content-Type: application/json" \
          -H "Authorization: Bearer $YOUR_GATEWAY_TOKEN" \
          -d '{
            "model": "priority-auto",
            "messages": [{"role": "user", "content": "Say this is a test!"}]
          }'
        ```
    *   **Checking Available Models:**
        ```bash
        curl http://localhost:12345/v1/models -H "Authorization: Bearer $YOUR_GATEWAY_TOKEN"
        ```

## Dynamic Scoring Mechanism

The LLM Gateway uses a dynamic scoring mechanism to prioritize and route requests to the most suitable provider. The score for each provider is calculated based on the following factors:

1.  **Priority**: A static priority value assigned to each provider for a specific model.
2.  **Leaderboard Score**: A score fetched from a leaderboard (e.g., BigCodeBench) that reflects the model's performance.
3.  **Parameter Score**: A score calculated based on the model's parameters (e.g., size like 7b, 8x7b).
4.  **Context Score**: A score calculated based on the context length (`context_window`) of the model.
5.  **Adjustments**: Scores defined in `config/model-adjustments.yaml` based on regex matching model names.
6.  **Provider Base Priority**: A base multiplier defined per provider in `config/meta-providers.yaml`.

The final score is a weighted sum of these factors, with weights defined by constants in `src/consts.py` (e.g., `MODEL_LEADERBOARD_SCORE_SCALAR`, `MODEL_PARAM_SCORE_SCALAR`, `MODEL_CTX_SCORE_SCALAR`, `MODEL_ADJUSTMENT_SCALAR`). Providers with an `overall_score` <= 0 are filtered out.

## Features

-   Single API endpoint matching OpenAI's specification.
-   **Bearer Token Authentication**: Secure access using configurable tokens via the `AUTH_TOKENS` environment variable.
-   Automatic failover to backup providers based on dynamic scoring.
-   Rate limit handling and cooldown management.
-   Priority-based routing using dynamic scoring.
-   Async requests for improved performance.
-   Health check endpoint (`/health`, `/ok` - no authentication required).
-   Statistics endpoint (`/stats` - requires authentication).
-   Models endpoint (`/models`, `/v1/models` - requires authentication) listing available and scored providers.
-   Robust error handling.

## Project Rules

*   Always update `pyproject.toml` (via `uv pip compile`) when adding/changing dependencies.
*   Keep the README updated with current status and configuration details.
*   Write tests for new features and ensure existing tests pass.
*   Maintain API compatibility with OpenAI's spec.
*   Document all configuration options clearly.
*   After any change, run the project locally (`uv run python src/main.py`) to ensure it starts and basic functionality works.
*   After any change, run the full test and lint suite (`uv run lint-test`) to ensure code quality and correctness.

## Development

1.  **Install Dev Dependencies:** `uv sync --dev`
2.  **Run with Auto-Reload:** `uv run python -m uvicorn src.main:app --reload --port 12345`
3.  **Run Linters & Tests:** Before committing or pushing, run the full check suite:
    ```bash
    uv run lint-test
    ```
    This command is defined in `pyproject.toml` and runs formatters (autopep8, black), type checkers (mypy), linters (flake8, pylint), and tests (pytest). Ensure it passes to avoid CI failures.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=n00b001/BEST&type=Date)](https://www.star-history.com/#n00b001/BEST&Date)

## Todo

*   Add tests for the authentication layer.
*   Add more specific examples for different client libraries.
*   Implement per-token rate limiting (optional enhancement).
*   Consider token revocation mechanism without service restart (optional enhancement).
