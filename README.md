# LLM Gateway

A unified API gateway for multiple OpenAI-compatible LLM providers. This project allows you to use various LLM services through a single, OpenAI-compatible API, simplifying integration and providing features like failover and rate limiting.


## Getting Started

1.  **Install Dependencies:** Install the required packages using either:
    -   `pip install -r requirements.txt`
    -   `uv sync`

2.  **Configure Providers:**
    -   Add your LLM providers to `config/meta-providers.yaml`.  See the [Configuration](#configuration) section for details.
    -   Set up your API keys in a `.env` file.  You can find resources for API keys here: [https://github.com/cheahjs/free-llm-api-resources](https://github.com/cheahjs/free-llm-api-resources)

3.  **Start the Service:**
    -   Run the gateway using: `uvicorn src.main:app`

4.  **Use the API:**
    -   Point your OpenAI client to the gateway:
        ```python
        import openai

        openai.api_base = "http://localhost:8000/v1"
        openai.api_key = "any-key"  # Not used by the gateway
        ```

## Configuration

The `config/meta-providers.yaml` file defines the available LLM providers and their configurations.  Here's an example:

```yaml
providers:
  - name: openai
    type: openai
    api_key: ${OPENAI_API_KEY}
    base_url: https://api.openai.com/v1
    priority: 1
    rate_limit:
      requests_per_minute: 60
  - name: cohere
    type: cohere
    api_key: ${COHERE_API_KEY}
    base_url: https://api.cohere.ai/v1
    priority: 2
    rate_limit:
      requests_per_minute: 30
```

-   `name`: A unique identifier for the provider.
-   `type`: The type of LLM provider (e.g., `openai`, `cohere`).
-   `api_key`: The API key for the provider.  Use environment variables for security.
-   `base_url`: The base URL for the provider's API.
-   `priority`: The priority of the provider (lower numbers are higher priority).  The gateway will try providers in order of priority.
-   `rate_limit`: Optional rate limiting configuration.

## Features

-   Single API endpoint matching OpenAI's specification.
-   Automatic failover to backup providers.  The gateway will automatically try the next available provider if the primary provider fails.
-   Rate limit handling and cooldown management.  The gateway will respect the rate limits of each provider.
-   Priority-based routing.
-   Async requests for improved performance.
-   Health check endpoint.
-   Robust error handling.

## Example Usage with Different LLM Providers

Here's how to use the gateway with different providers:

```python
import openai

# Using OpenAI (configured in meta-providers.yaml)
openai.api_base = "http://localhost:8000/v1"
openai.api_key = "any-key"
response = openai.Completion.create(
    engine="text-davinci-003",
    prompt="Translate the following English text to French: Hello, world!",
    max_tokens=50,
)
print(response.choices[0].text)

# Using Cohere (configured in meta-providers.yaml)
# The gateway will automatically route the request to Cohere if it's configured and available.
# You don't need to change your code.
```

## Monitoring and Logging

Consider using tools like Prometheus and Grafana for monitoring the gateway's performance.  You can also use a logging library like `loguru` or the built-in Python `logging` module to log requests, responses, and errors.

## Project Rules

-   Always update requirements.txt when adding dependencies
-   Keep the README updated with current status
-   Write tests for new features
-   Maintain API compatibility with OpenAI's spec
-   Document all configuration options
-   After any change, run the project to make sure it still works
-   After any change, run any tests to make sure they still pass

## This implementation provides:

1.  OpenAI-compatible API endpoint
2.  Priority-based routing with failover
3.  Rate limit handling with Retry-After parsing
4.  Async requests for better performance
5.  Clean configuration management
6.  Health check endpoint
7.  Proper error handling

## To use it:

1.  Configure your LLM providers in `config/meta-providers.yaml`
2.  Setup up your API keys in `.env`
3.  Start the service with `uvicorn src.main:app`
4.  Use it like you would the OpenAI API:

```python
import openai

openai.api_base = "http://localhost:8000/v1"
openai.api_key = "any-key"  # Not used by gateway
```

## To develop:

1.  `uv sync --dev`
2.  `uvicorn src.main:app --reload`
3.  And after making code changes, run this before pushing (if it fails here, it will fail github actions) `uv run python -m autopep8 --exclude .venv -ri . && uv run python -m black --fast --color -l 120 . && uv run python -m mypy --exclude .venv --follow-untyped-imports --explicit-package-bases . && uv run python -m flake8 --exclude .venv --max-line-length 120 . && uv run python -m pylint --ignore .venv --output-format=colorized \
              --max-line-length 120 --fail-under 5 --fail-on E . && uv run python -m pytest --color yes --verbosity=3;`

## Todo

-   Add more concrete examples to the documentation.

## Contributing
## Future Enhancements

- **Support for more LLM providers:**  Expand the gateway to support a wider range of LLM providers, including open-source models.
- **Improved rate limiting:** Implement more sophisticated rate limiting strategies, such as token-based rate limiting.
- **Caching:** Add caching to reduce latency and improve performance.
- **Request transformation:** Allow users to transform requests before sending them to the LLM provider.
- **Response transformation:** Allow users to transform responses before returning them to the client.
- **UI:** Develop a user interface for managing the gateway.
## Future Enhancements

- **Support for more LLM providers:**  Expand the gateway to support a wider range of LLM providers, including open-source models.
- **Improved rate limiting:** Implement more sophisticated rate limiting strategies, such as token-based rate limiting.
- **Caching:** Add caching to reduce latency and improve performance.
- **Request transformation:** Allow users to transform requests before sending them to the LLM provider.
- **Response transformation:** Allow users to transform responses before returning them to the client.
- **UI:** Develop a user interface for managing the gateway.

## Contributing

Contributions to this project are welcome! Please follow these guidelines:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and commit them with clear messages.
4.  Write tests for your changes.
5.  Run the tests and ensure they pass.
6.  Submit a pull request.
## Future Enhancements

- **Support for more LLM providers:**  Expand the gateway to support a wider range of LLM providers, including open-source models.
- **Improved rate limiting:** Implement more sophisticated rate limiting strategies, such as token-based rate limiting.
- **Caching:** Add caching to reduce latency and improve performance.
- **Request transformation:** Allow users to transform requests before sending them to the LLM provider.
- **Response transformation:** Allow users to transform responses before returning them to the client.
- **UI:** Develop a user interface for managing the gateway.
## Future Enhancements

- **Support for more LLM providers:**  Expand the gateway to support a wider range of LLM providers, including open-source models.
- **Improved rate limiting:** Implement more sophisticated rate limiting strategies, such as token-based rate limiting.
- **Caching:** Add caching to reduce latency and improve performance.
- **Request transformation:** Allow users to transform requests before sending them to the LLM provider.
- **Response transformation:** Allow users to transform responses before returning them to the client.
- **UI:** Develop a user interface for managing the gateway.
## Future Enhancements

- **Support for more LLM providers:**  Expand the gateway to support a wider range of LLM providers, including open-source models.
- **Improved rate limiting:** Implement more sophisticated rate limiting strategies, such as token-based rate limiting.
- **Caching:** Add caching to reduce latency and improve performance.
- **Request transformation:** Allow users to transform requests before sending them to the LLM provider.
- **Response transformation:** Allow users to transform responses before returning them to the client.
- **UI:** Develop a user interface for managing the gateway.
## Future Enhancements

- **Support for more LLM providers:**  Expand the gateway to support a wider range of LLM providers, including open-source models.
- **Improved rate limiting:** Implement more sophisticated rate limiting strategies, such as token-based rate limiting.
- **Caching:** Add caching to reduce latency and improve performance.
- **Request transformation:** Allow users to transform requests before sending them to the LLM provider.
- **Response transformation:** Allow users to transform responses before returning them to the client.
- **UI:** Develop a user interface for managing the gateway.

Contributions to this project are welcome! Please follow these guidelines:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and commit them with clear messages.
4.  Write tests for your changes.
5.  Run the tests and ensure they pass.
6.  Submit a pull request.
1.  **Install Dependencies:** Install the required packages using either:
    -   `pip install -r requirements.txt`
    -   `uv sync`

2.  **Configure Providers:**
    -   Add your LLM providers to `config/meta-providers.yaml`.  See the [Configuration](#configuration) section for details.
    -   Set up your API keys in a `.env` file.  You can find resources for API keys here: [https://github.com/cheahjs/free-llm-api-resources](https://github.com/cheahjs/free-llm-api-resources)

3.  **Start the Service:**
    -   Run the gateway using: `uvicorn src.main:app`

4.  **Use the API:**
    -   Point your OpenAI client to the gateway:
        ```python
        import openai

        openai.api_base = "http://localhost:8000/v1"
        openai.api_key = "any-key"  # Not used by the gateway
        ```

## Configuration

The `config/meta-providers.yaml` file defines the available LLM providers and their configurations.  Here's an example:

```yaml
providers:
  - name: openai
    type: openai
    api_key: ${OPENAI_API_KEY}
    base_url: https://api.openai.com/v1
    priority: 1
    rate_limit:
      requests_per_minute: 60
  - name: cohere
    type: cohere
    api_key: ${COHERE_API_KEY}
    base_url: https://api.cohere.ai/v1
    priority: 2
    rate_limit:
      requests_per_minute: 30
```

-   `name`: A unique identifier for the provider.
-   `type`: The type of LLM provider (e.g., `openai`, `cohere`).
-   `api_key`: The API key for the provider.  Use environment variables for security.
-   `base_url`: The base URL for the provider's API.
-   `priority`: The priority of the provider (lower numbers are higher priority).  The gateway will try providers in order of priority.
-   `rate_limit`: Optional rate limiting configuration.

## Features

-   Single API endpoint matching OpenAI's specification.
-   Automatic failover to backup providers.  The gateway will automatically try the next available provider if the primary provider fails.
-   Rate limit handling and cooldown management.  The gateway will respect the rate limits of each provider.
-   Priority-based routing.
-   Async requests for improved performance.
-   Health check endpoint.
-   Robust error handling.

## Example Usage with Different LLM Providers

Here's how to use the gateway with different providers:

```python
import openai

# Using OpenAI (configured in meta-providers.yaml)
openai.api_base = "http://localhost:8000/v1"
openai.api_key = "any-key"
response = openai.Completion.create(
    engine="text-davinci-003",
    prompt="Translate the following English text to French: Hello, world!",
    max_tokens=50,
)
print(response.choices[0].text)

# Using Cohere (configured in meta-providers.yaml)
# The gateway will automatically route the request to Cohere if it's configured and available.
# You don't need to change your code.
```

## Monitoring and Logging

Consider using tools like Prometheus and Grafana for monitoring the gateway's performance.  You can also use a logging library like `loguru` or the built-in Python `logging` module to log requests, responses, and errors.

## Project Rules

-   Always update requirements.txt when adding dependencies
-   Keep the README updated with current status
-   Write tests for new features
-   Maintain API compatibility with OpenAI's spec
-   Document all configuration options
-   After any change, run the project to make sure it still works
-   After any change, run any tests to make sure they still pass

## This implementation provides:

1.  OpenAI-compatible API endpoint
2.  Priority-based routing with failover
3.  Rate limit handling with Retry-After parsing
4.  Async requests for better performance
5.  Clean configuration management
6.  Health check endpoint
7.  Proper error handling

## To use it:

1.  Configure your LLM providers in `config/meta-providers.yaml`
2.  Setup up your API keys in `.env`
3.  Start the service with `uvicorn src.main:app`
4.  Use it like you would the OpenAI API:

```python
import openai

openai.api_base = "http://localhost:8000/v1"
openai.api_key = "any-key"  # Not used by gateway
```

## To develop:

1.  `uv sync --dev`
2.  `uvicorn src.main:app --reload`
3.  And after making code changes, run this before pushing (if it fails here, it will fail github actions) `uv run python -m autopep8 --exclude .venv -ri . && uv run python -m black --fast --color -l 120 . && uv run python -m mypy --exclude .venv --follow-untyped-imports --explicit-package-bases . && uv run python -m flake8 --exclude .venv --max-line-length 120 . && uv run python -m pylint --ignore .venv --output-format=colorized \
              --max-line-length 120 --fail-under 5 --fail-on E . && uv run python -m pytest --color yes --verbosity=3;`

## Todo

-   Add more concrete examples to the documentation.

1. **Install Dependencies:** Install the required packages using either:
    - `pip install -r requirements.txt`
    - `uv sync`

2. **Configure Providers:**
    -  Add your LLM providers to `config/meta-providers.yaml`.
    -  Set up your API keys in a `.env` file.  You can find resources for API keys here: [https://github.com/cheahjs/free-llm-api-resources](https://github.com/cheahjs/free-llm-api-resources)

3. **Start the Service:**
    - Run the gateway using: `uvicorn src.main:app`

4. **Use the API:**
   -  Point your OpenAI client to the gateway:
    ```python
    import openai

    openai.api_base = "http://localhost:8000/v1"
    openai.api_key = "any-key"  # Not used by the gateway
    ```

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
3. Start the service with `uvicorn src.main:app`
4. Use it like you would the OpenAI API:

```python
import openai

openai.api_base = "http://localhost:8000/v1"
openai.api_key = "any-key"  # Not used by gateway
```

## To develop:

1. `uv sync --dev`
2. `uvicorn src.main:app --reload`
3. And after making code changes, run this before pushing (if it fails here, it will fail github actions) `uv run python -m autopep8 --exclude .venv -ri . && uv run python -m black --fast --color -l 120 . && uv run python -m mypy --exclude .venv --follow-untyped-imports --explicit-package-bases . && uv run python -m flake8 --exclude .venv --max-line-length 120 . && uv run python -m pylint --ignore .venv --output-format=colorized \
          --max-line-length 120 --fail-under 5 --fail-on E . && uv run python -m pytest --color yes --verbosity=3;`

## Todo

- Add more concrete examples to the documentation.
