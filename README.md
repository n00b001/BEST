Where to get API keys:
https://github.com/cheahjs/free-llm-api-resources

# LLM Gateway

A unified API gateway for multiple OpenAI-compatible LLM providers.

## Features

- Single API endpoint matching OpenAI's specification
- Automatic failover to backup providers
- Rate limit handling and cooldown management
- Priority-based routing

## Setup

1. Install dependencies (do one or the other):
    1. `pip install -r requirements.txt`
    2. `uv sync`

## TODO

1. Add authentication
2. Implement request caching
3. Add prometheus metrics
4. Implement circuit breaker pattern

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