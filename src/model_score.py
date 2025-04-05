import io
import logging
from functools import lru_cache
from typing import Any, Dict, List
from urllib.request import urlopen

import numpy as np
import pandas as pd
import requests
import yaml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global constants and session configuration
HEADERS = {"User-Agent": "Mozilla/5.0"}
SESSION = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS"],
)
adapter = HTTPAdapter(max_retries=retry_strategy)
SESSION.mount("https://", adapter)
SESSION.mount("http://", adapter)
SESSION.headers.update(HEADERS)


def normalize_column(source_df: pd.DataFrame, target_df: pd.DataFrame, column: str) -> None:
    """
    Normalizes the values of a column in source_df and assigns them to target_df.
    """
    target_df[column] = (source_df[column] - source_df[column].min()) / (
        source_df[column].max() - source_df[column].min()
    )


@lru_cache(maxsize=1)
def fetch_bigcodebench_leaderboard_easy() -> pd.DataFrame:
    """
    Fetches and processes the 'easy' BigCodeBench leaderboard data.
    """
    url = "https://bigcode-bench.github.io/results.json"
    df1 = pd.read_json(url)
    new_rows = []

    for col in df1.columns:
        try:
            metrics = df1[col]["pass@1"]
            if isinstance(metrics, dict):
                if metrics.get("instruct") is not None:
                    new_rows.append({"model_name": f"{col}_instruct", "score": metrics["instruct"]})
                if metrics.get("complete") is not None:
                    new_rows.append({"model_name": f"{col}_complete", "score": metrics["complete"]})
            else:
                logger.warning("Unexpected format for metrics in column '%s', metrics: %s", col, metrics)
        except (SyntaxError, NameError, TypeError) as e:
            logger.warning("Error processing data in column '%s': %s", col, e)
        except KeyError as e:
            logger.warning("Key %s not found for column '%s'", e, col)

    new_df = pd.DataFrame(new_rows)
    new_df["score"] = (new_df["score"] - new_df["score"].min()) / (new_df["score"].max() - new_df["score"].min())
    return new_df


@lru_cache(maxsize=1)
def fetch_bigcodebench_leaderboard_hard() -> pd.DataFrame:
    """
    Fetches and processes the 'hard' BigCodeBench leaderboard data.
    """
    url = "https://bigcode-bench.github.io/results-hard.json"
    df1 = pd.read_json(url)
    new_rows = []

    for col in df1.columns:
        try:
            metrics = df1[col]["pass@1"]
            if isinstance(metrics, dict):
                if metrics.get("instruct") is not None:
                    new_rows.append({"model_name": f"{col}_instruct", "score": metrics["instruct"]})
                if metrics.get("complete") is not None:
                    new_rows.append({"model_name": f"{col}_complete", "score": metrics["complete"]})
            else:
                logger.warning("Unexpected format for metrics in column '%s', metrics: %s", col, metrics)
        except (SyntaxError, NameError, TypeError) as e:
            logger.warning("Error processing data in column '%s': %s", col, e)
        except KeyError as e:
            logger.warning("Key %s not found for column '%s'", e, col)

    new_df = pd.DataFrame(new_rows)
    new_df["score"] = (new_df["score"] - new_df["score"].min()) / (new_df["score"].max() - new_df["score"].min())
    return new_df


@lru_cache(maxsize=1)
def fetch_evalplus_leaderboard2() -> pd.DataFrame:
    """
    Fetches EvalPlus leaderboard data and computes the mean of various scores.
    """
    url = "https://evalplus.github.io/results.json"
    df = pd.read_json(url)
    new_rows = []

    for model_name, model_data in df.items():
        try:
            pass_at_1 = model_data["pass@1"]
            if isinstance(pass_at_1, dict):
                new_rows.append(
                    {
                        "model_name": model_name,
                        "humaneval_score": pass_at_1["humaneval"],
                        "humaneval+_score": pass_at_1["humaneval+"],
                        "mbpp_score": pass_at_1["mbpp"],
                        "mbpp+_score": pass_at_1["mbpp+"],
                    }
                )
        except KeyError:
            logger.warning("Skipping model %s: 'pass@1' key not found", model_name)

    new_df = pd.DataFrame(new_rows)
    for col in ["humaneval_score", "humaneval+_score", "mbpp_score", "mbpp+_score"]:
        normalize_column(new_df, new_df, col)
    new_df["score"] = new_df[["humaneval_score", "humaneval+_score", "mbpp_score", "mbpp+_score"]].mean(axis=1)
    return new_df


@lru_cache(maxsize=1)
def fetch_crux_leaderboard2() -> pd.DataFrame:
    """
    Fetches Crux leaderboard data from CSV and processes it.
    """
    url = "https://crux-eval.github.io/data.csv"
    df_crux = pd.read_csv(url)
    df_crux = df_crux.replace("-", np.nan)
    df_crux[["i@1", "i@5", "o@1", "o@5"]] = df_crux[["i@1", "i@5", "o@1", "o@5"]].astype(float)

    new_df = pd.DataFrame()
    new_df["model_name"] = df_crux["Model"]
    for col in ["i@1", "i@5", "o@1", "o@5"]:
        normalize_column(df_crux, new_df, col)
    new_df["score"] = new_df[["i@1", "i@5", "o@1", "o@5"]].mean(axis=1)
    return new_df


@lru_cache(maxsize=1)
def fetch_tabby_leaderboard2() -> pd.DataFrame:
    """
    Fetches Tabby leaderboard data from YAML, normalizes the scores, and calculates the mean score.
    """
    url = "https://leaderboard.tabbyml.com/tabby.yml"
    try:
        with urlopen(url) as file:
            data = yaml.safe_load(file)
    except Exception as e:
        logger.error("An error occurred while fetching Tabby data: %s", e)
        data = {}

    rows = []
    for model_name, model_data in data.items():
        for metric, scores in model_data.items():
            row = {"model_name": f"{model_name}_{metric.lower()}"}
            for lang, score in scores.items():
                row[f"{lang.lower()}_score"] = score
            rows.append(row)

    df = pd.DataFrame(rows)
    score_columns = [col for col in df.columns if col.endswith("_score")]
    for col in score_columns:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    df["score"] = df[score_columns].mean(axis=1)
    df.sort_values(by="score", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


@lru_cache(maxsize=1)
def fetch_aider_leaderboard2() -> pd.DataFrame:
    """
    Fetches Aider leaderboard data, processes both old and new formats, and merges them.
    """
    url = "https://aider.chat/assets/js/search-data.json"
    df = pd.read_json(url).T

    old_leaderboard = df[df["title"] == "Code editing leaderboard"]["content"]
    new_leaderboard = df[df["title"] == "Polyglot leaderboard"]["content"]

    new_leaderboard = new_leaderboard.to_list()[0]
    old_leaderboard = old_leaderboard.to_list()[0]

    new_leaderboard = "\n".join(new_leaderboard.split("intervention. ")[-1].split(" . "))
    old_leaderboard = "\n".join(old_leaderboard.split("intervention. ")[-1].split(" . "))

    old_buffer = io.StringIO(old_leaderboard)
    new_buffer = io.StringIO(new_leaderboard)

    old_df = pd.read_csv(old_buffer, sep="|").rename(columns=lambda x: x.strip())
    new_df = pd.read_csv(new_buffer, sep="|").rename(columns=lambda x: x.strip())

    def process_leaderboard(
        df: pd.DataFrame, model_col: str, edit_format_col: str, percent_col: str, format_col: str
    ) -> List[Dict[str, Any]]:
        rows = []
        for _, row in df.iterrows():
            try:
                model_name = f"{row[model_col].strip()}_{row[edit_format_col].strip()}"
                percent_completed = float(row[percent_col].replace("%", ""))
                percent_format = float(row[format_col].replace("%", ""))
                score = (percent_completed + percent_format) / 200
                rows.append({"model_name": model_name, "score": score})
            except (ValueError, AttributeError, KeyError):
                continue
        return rows

    old_rows = process_leaderboard(
        old_df, "Model", "Edit format", "Percent completed correctly", "Percent using correct edit format"
    )
    new_rows = process_leaderboard(
        new_df, "Model", "Edit format", "Percent correct", "Percent using correct edit format"
    )

    old_processed = pd.DataFrame(old_rows)
    new_processed = pd.DataFrame(new_rows)

    merged_df = pd.merge(old_processed, new_processed, on="model_name", how="outer", validate="many_to_many")
    merged_df["score"] = merged_df.apply(
        lambda row: (
            np.mean([row["score_x"], row["score_y"]])
            if pd.notna(row["score_x"]) and pd.notna(row["score_y"])
            else row["score_x"] if pd.notna(row["score_x"]) else row["score_y"]
        ),
        axis=1,
    )
    merged_df.drop(["score_x", "score_y"], axis=1, inplace=True, errors="ignore")
    return merged_df


def merge_dataframes(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Merges multiple dataframes of leaderboard scores into one, averaging scores for duplicate model names.
    """
    merged_df = pd.DataFrame(columns=["model_name", "score"])
    for df in dfs:
        for _, row in df.iterrows():
            model_name = row["model_name"]
            row_score = row["score"]
            if model_name in merged_df["model_name"].values:
                existing_score = merged_df.loc[merged_df["model_name"] == model_name, "score"].astype(float)
                merged_df.loc[merged_df["model_name"] == model_name, "score"] = (existing_score + row_score) / 2
            else:
                merged_df = pd.concat(
                    [merged_df, pd.DataFrame({"model_name": [model_name], "score": [row_score]})],
                    ignore_index=True,
                )
    return merged_df


def get_leaderboard_score(model_name: str) -> float:
    """
    Aggregates scores across all tests and leaderboards using normalized averaging.
    """
    merged_df = get_merged_leaderboard_df()
    scores = merged_df[merged_df["model_name"].str.contains(model_name)]
    if scores.empty:
        return 0.0
    return float(scores["score"].mean())


@lru_cache(maxsize=1)
def get_merged_leaderboard_df() -> pd.DataFrame:
    """
    Merges all leaderboard data sources into a single sorted dataframe.
    """
    sources = [
        fetch_bigcodebench_leaderboard_easy(),
        fetch_bigcodebench_leaderboard_hard(),
        fetch_evalplus_leaderboard2(),
        fetch_crux_leaderboard2(),
        fetch_tabby_leaderboard2(),
        fetch_aider_leaderboard2(),
    ]
    merged_df = merge_dataframes(sources)
    merged_df.sort_values(by="score", ascending=False, inplace=True)
    merged_df.reset_index(drop=True, inplace=True)
    return merged_df


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
        if score:
            model_scores.append((model, score))
        else:
            logger.info("No data found for %s", model)

    # Sort models by score in descending order
    model_scores.sort(key=lambda x: x[1], reverse=True)

    # Print ranked results
    print("Rankings:")
    for rank, (model, score) in enumerate(model_scores, 1):
        print(f"Rank {rank}: {model} - Aggregated Score: {score:.2f}")
