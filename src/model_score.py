import csv
def fetch_leaderboard(urls: List[str], parser: callable) -> Dict[str, Dict[str, List[float]]]:
    """Fetches and processes leaderboard data."""
    scores: Dict[str, Dict[str, List[float]]] = {}
    for url in urls:
        data = fetch_json(url)
        if not data:
            continue
        scores.update(parser(data))
    return scores
import io
import logging
import re
from collections import defaultdict
from functools import lru_cache
from typing import Any, Dict, List, Optional, Generator, Tuple, DefaultDict

import requests
import yaml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HEADERS = {"User-Agent": "Mozilla/5.0"}
SESSION = requests.Session()

# Configure retries for transient errors
retry_strategy = Retry(
    total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504], allowed_methods=["HEAD", "GET", "OPTIONS"]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
SESSION.mount("https://", adapter)
SESSION.mount("http://", adapter)
SESSION.headers.update(HEADERS)


def normalize_model_name(name: str) -> str:
    """Normalizes a model name by stripping whitespace, converting to lowercase,
    and removing non-alphanumeric characters."""
    name = name.strip().lower()
    return re.sub(r"[^a-z0-9]", "", name)


def safe_float(value: Any) -> float:
    """
    Safely converts a value to float. Handles:
    - None, hyphens, and empty strings as 0.0
    - Numeric types directly
    - Strings that can be converted to float
    Raises ValueError for non-convertible values.
    """
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)

    s = str(value).strip()
    if s in ("-", ""):
        return 0.0

    try:
        return float(s)
    except ValueError:
        raise ValueError(f"Could not convert '{value}' to float")


def fetch_json(url: str, timeout: int = 10) -> Optional[Any]:
    """Fetches JSON data from the provided URL with retries and error handling."""
    try:
        response = SESSION.get(url, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error fetching JSON from {url}: {e}")
        return None


def fetch_text(url: str, timeout: int = 10) -> Optional[str]:
    """Fetches text data from the provided URL with retries and error handling."""
    try:
        response = SESSION.get(url, timeout=timeout)
        response.raise_for_status()
        return response.text
    except Exception as e:
        logger.error(f"Error fetching text from {url}: {e}")
        return None


def fetch_bigcodebench_leaderboard() -> Dict[str, Dict[str, List[float]]]:
    """Fetches scores from BigCodeBench leaderboards."""
    def parse_data(data: Any) -> Dict[str, Dict[str, List[float]]]:
        scores = {}
        for model_name, model_data in data.items():
            normalized = normalize_model_name(model_name)
            pass_at1 = model_data.get("pass@1", {})
            try:
                instruct = safe_float(pass_at1.get("instruct", 0))
                complete = safe_float(pass_at1.get("complete", 0))
                if normalized not in scores:
                    scores[normalized] = {}
                scores[normalized].setdefault("instruct", []).append(instruct)
                scores[normalized].setdefault("complete", []).append(complete)
            except Exception as e:
                logger.warning(f"Skipping {model_name}: {e}")
        return scores

    return fetch_leaderboard(
        urls=["https://bigcode-bench.github.io/results.json", "https://bigcode-bench.github.io/results-hard.json"],
        parser=parse_data
    )


@lru_cache(maxsize=1)
def fetch_evalplus_leaderboard() -> Dict[str, Dict[str, List[float]]]:
    """Fetches scores from EvalPlus leaderboard."""
    url = "https://evalplus.github.io/results.json"
    scores: Dict[str, Dict[str, List[float]]] = {}
    data = fetch_json(url)
    if not data:
        return scores
    for model_name, model_data in data.items():
        normalized = normalize_model_name(model_name)
        pass_at1 = model_data.get("pass@1", {})
        for key in ["humaneval", "humaneval+", "mbpp", "mbpp+"]:
            val = pass_at1.get(key)
            if val is None:
                continue
            try:
                score = safe_float(val)
                if normalized not in scores:
                    scores[normalized] = {}
                scores[normalized].setdefault(key, []).append(score)
            except Exception as e:
                logger.warning(f"Skipping {model_name} {key}: {e}")
    return scores


@lru_cache(maxsize=1)
def fetch_crux_leaderboard() -> Dict[str, Dict[str, List[float]]]:
    """Fetches and processes Crux leaderboard from CSV data."""
    url = "https://crux-eval.github.io/data.csv"
    scores: Dict[str, Dict[str, List[float]]] = {}
    text = fetch_text(url)
    if not text:
        return scores
    try:
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            if not (model_name := row.get("Model")):
                continue
            normalized = normalize_model_name(model_name)
            for col in ["i@1", "i@5", "o@1", "o@5"]:
                val = row.get(col, 0)
                try:
                    score = safe_float(val)
                    if normalized not in scores:
                        scores[normalized] = {}
                    scores[normalized].setdefault(col, []).append(score)
                except Exception as e:
                    logger.warning(f"Skipping {model_name} {col}: {e}")
    except Exception as e:
        logger.error(f"Error processing Crux data: {e}")
    return scores


@lru_cache(maxsize=1)
def fetch_tabby_leaderboard() -> Dict[str, Dict[str, List[float]]]:
    """Fetches and processes scores from Tabby YAML leaderboard."""
    url = "https://leaderboard.tabbyml.com/tabby.yml"
    scores: DefaultDict = defaultdict(lambda: defaultdict(list))
    text = fetch_text(url)

    if not text:
        return scores

    try:
        data = yaml.safe_load(text)
        for model_name, methods in data.items():
            normalized = normalize_model_name(model_name)
            if isinstance(methods, dict):
                for test_name, score_value in _generate_test_scores(methods):
                    try:
                        score = safe_float(score_value)
                        scores[normalized][test_name].append(score)
                    except Exception as e:
                        logger.warning(f"Skipping {model_name} {test_name}: {e}")
    except Exception as e:
        logger.error(f"Error processing Tabby data: {e}")

    return scores


def _generate_test_scores(methods: dict) -> Generator[Tuple[str, Any], None, None]:
    """Generates (test_name, score_value) pairs from methods data."""
    for method, lang_scores in methods.items():
        if isinstance(lang_scores, dict):
            for lang, score_value in lang_scores.items():
                yield f"{method}_{lang}", score_value


@lru_cache(maxsize=1)
def fetch_aider_leaderboard() -> Dict[str, Dict[str, List[float]]]:
    """Fetches and processes scores from Aider leaderboard."""
    url = "https://aider.chat/assets/js/search-data.json"
    scores: Dict[str, Dict[str, List[float]]] = {}
    data = fetch_json(url)
    if not data:
        return scores

    # Find the correct leaderboard entry
    entry = next(
        (
            e
            for e in data.values()
            if e.get("doc") == "Aider LLM Leaderboards" and "Polyglot leaderboard" in e.get("title", "")
        ),
        None,
    )
    if not entry:
        logger.error("Aider leaderboard entry not found")
        return scores

    # Extract and process scores
    content = entry.get("content", "")
    matches = re.findall(r"([^\|]+?)\s*\|\s*([\d\.]+)%\s*\|\s*([\d\.]+)%", content)
    for model_name, score1_str, score2_str in matches:
        try:
            # Convert percentage to decimal
            score1 = safe_float(score1_str) / 100
            score2 = safe_float(score2_str) / 100
            normalized = normalize_model_name(model_name)
            if normalized not in scores:
                scores[normalized] = {}
            scores[normalized].setdefault("aider_score1", []).append(score1)
            scores[normalized].setdefault("aider_score2", []).append(score2)
        except Exception as e:
            logger.warning(f"Skipping {model_name}: {e}")
    return scores


def get_leaderboard_score(model_name: str) -> float:
    """Aggregates scores across all tests and leaderboards using normalized averaging."""
    normalized = normalize_model_name(model_name)
    leaderboard_averages = []

    sources = [
        fetch_bigcodebench_leaderboard(),
        fetch_evalplus_leaderboard(),
        fetch_crux_leaderboard(),
        fetch_tabby_leaderboard(),
        fetch_aider_leaderboard(),
    ]

    for source in sources:
        leaderboard_scores = []
        # Find all matching models in this leaderboard
        for leaderboard_model in source:
            if normalized in leaderboard_model:
                # Collect all scores from all tests for this model entry
                for test_scores in source[leaderboard_model].values():
                    leaderboard_scores.extend(test_scores)

        if leaderboard_scores:
            # Calculate average for this leaderboard
            leaderboard_avg = sum(leaderboard_scores) / len(leaderboard_scores)
            leaderboard_averages.append(leaderboard_avg)

    if not leaderboard_averages:
        return 0

    # Calculate final score as average of leaderboard averages
    final_score = sum(leaderboard_averages) / len(leaderboard_averages)
    return final_score


if __name__ == "__main__":
    models = [
        "Magicoder-S-DS-6.7B",
        "StarCoder2-15B-Instruct-v0.1",
        "OpenCoder-8B-Instruct",
        "phi-1",
        "DeepSeekCoder-V2-Lite",
        "claude-3-5-sonnet-20241022",
        "phi-1.5",
        "KwaiCoder-23B-A4B-v1",
        "Qwen-QWQ",
        "GPT-4",
        "GPT-4o",
        "GPT-3.5",
    ]
    model_scores = []
    for model in models:
        score = get_leaderboard_score(model)
        if score is not None:
            model_scores.append((model, score))
        else:
            print(f"No data found for {model}")
    # Sort models by score in descending order
    model_scores.sort(key=lambda x: x[1], reverse=True)
    # Print ranked results
    print("Rankings:")
    for rank, (model, score) in enumerate(model_scores, 1):
        print(f"Rank {rank}: {model} - Aggregated Score: {score:.2f}")
