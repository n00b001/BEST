import csv
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
    name = name.split("/")[-1].strip().lower()
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


@lru_cache(maxsize=1)
def fetch_bigcodebench_leaderboard() -> Dict[str, Dict[str, List[float]]]:
    """Fetches scores from BigCodeBench leaderboards."""
    urls = [
        "https://bigcode-bench.github.io/results.json",
        "https://bigcode-bench.github.io/results-hard.json",
    ]
    scores: Dict[str, Dict[str, List[float]]] = {}
    for url in urls:
        data = fetch_json(url)
        if not data:
            continue
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


@lru_cache(maxsize=1)
def fetch_bigcodebench_leaderboard_easy():
    # prompt: using df1 as a data source, please create a new df:
    # it should have columns:
    # model_name, pass@1
    # Please use the column names from df1 as model names. Each column name in df1 is a the name of a model
    # please then extract the row "pass@1" (which is a string, example: "{'instruct': 36.2, 'complete': 47.6}")
    # into relevant rows.
    # For example, the column: "Magicoder-S-DS-6.7B" from df1 would be extracted from df1.
    # We would then append "_instruct" to the model name, and add it to the new dataframe:
    # model_name = "Magicoder-S-DS-6.7B_instruct"
    # pass@1 = 36.2
    # etc

    import pandas as pd

    new_rows = []
    df1 = pd.read_json("https://bigcode-bench.github.io/results.json")
    for col in df1.columns:
        try:
            metrics = df1[col]["pass@1"]
            if isinstance(metrics, dict):
                if "instruct" in metrics and metrics["instruct"] is not None:
                    new_rows.append({"model_name": f"{col}_instruct", "score": metrics["instruct"]})
                if "complete" in metrics and metrics["complete"] is not None:
                    new_rows.append({"model_name": f"{col}_complete", "score": metrics["complete"]})
            else:
                print(f"Warning: Unexpected format for metrics in column '{col}', metrics: {metrics}")
        except (SyntaxError, NameError, TypeError) as e:
            print(f"Warning: Error processing data in column '{col}', row: {df1[col]}")
            print(f"Exception: {e}")
        except KeyError as e:
            print(f"Warning: Key {e} not found for metrics in column '{col}', row: {df1[col]}")

    new_df = pd.DataFrame(new_rows)
    new_df["score"] = (new_df["score"] - new_df["score"].min()) / (new_df["score"].max() - new_df["score"].min())
    return new_df


def normalise(field, df, new_df):
    new_df[field] = (df[field] - df[field].min()) / (df[field].max() - df[field].min())


@lru_cache(maxsize=1)
def fetch_bigcodebench_leaderboard_hard():
    # prompt: using df1 as a data source, please create a new df:
    # it should have columns:
    # model_name, pass@1
    # Please use the column names from df1 as model names. Each column name in df1 is a the name of a model
    # please then extract the row "pass@1" (which is a string, example: "{'instruct': 36.2, 'complete': 47.6}")
    # into relevant rows.
    # For example, the column: "Magicoder-S-DS-6.7B" from df1 would be extracted from df1.
    # We would then append "_instruct" to the model name, and add it to the new dataframe:
    # model_name = "Magicoder-S-DS-6.7B_instruct"
    # pass@1 = 36.2
    # etc

    import pandas as pd

    # Assuming df1 is already defined as in the provided code

    new_rows = []
    df1 = pd.read_json("https://bigcode-bench.github.io/results-hard.json")
    for col in df1.columns:
        try:
            metrics = df1[col]["pass@1"]
            if isinstance(metrics, dict):
                if "instruct" in metrics and metrics["instruct"] is not None:
                    new_rows.append({"model_name": f"{col}_instruct", "score": metrics["instruct"]})
                if "complete" in metrics and metrics["complete"] is not None:
                    new_rows.append({"model_name": f"{col}_complete", "score": metrics["complete"]})
            else:
                print(f"Warning: Unexpected format for metrics in column '{col}', metrics: {metrics}")
        except (SyntaxError, NameError, TypeError) as e:
            print(f"Warning: Error processing data in column '{col}', row: {df1[col]}")
            print(f"Exception: {e}")
        except KeyError as e:
            print(f"Warning: Key {e} not found for metrics in column '{col}', row: {df1[col]}")

    new_df = pd.DataFrame(new_rows)
    new_df["score"] = (new_df["score"] - new_df["score"].min()) / (new_df["score"].max() - new_df["score"].min())
    return new_df


@lru_cache(maxsize=1)
def fetch_evalplus_leaderboard2():
    # prompt: Starting with 'df = pd.read_json("https://evalplus.github.io/results.json")'
    # Please create a new df with columns: model_name, score
    # here is the datamodel of the json file: "
    # {
    #       "OpenCoder-8B-Instruct": {
    #             "link": "https://huggingface.co/infly/OpenCoder-8B-Instruct",
    #             "open-data": "NONE",
    #             "pass@1": {
    #                   "humaneval": 81.7,
    #                   "humaneval+": 77.4,
    #                   "mbpp": 82,
    #                   "mbpp+": 71.4
    #             },
    #             "prompted": true,
    #             "size": 8
    #       },
    # "
    # Please mean average all the scores under pass@1 and assign the result to 'score' in the new DF
    # the model name can be found as "OpenCoder-8B-Instruct" in the exacmple (it will be the column in the df)
    import pandas as pd

    df = pd.read_json("https://evalplus.github.io/results.json")

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
            print(f"Skipping model {model_name}: 'pass@1' key not found or invalid data format.")

    new_df = pd.DataFrame(new_rows)
    normalise("humaneval_score", new_df, new_df)
    normalise("humaneval+_score", new_df, new_df)
    normalise("mbpp_score", new_df, new_df)
    normalise("mbpp+_score", new_df, new_df)
    new_df["score"] = new_df[["humaneval_score", "humaneval+_score", "mbpp_score", "mbpp+_score"]].mean(axis=1)
    return new_df


@lru_cache(maxsize=1)
def fetch_crux_leaderboard2():
    # prompt: Now we will do the same, this time using: "https://crux-eval.github.io/data.csv"
    # Please read this data, and create a new dataframe with columns: "model_name", "score"
    # Here is the data model for the CSV:
    # "
    # Model,i@1,i@5,o@1,o@5,link
    # phi-1,13.1,21.1,21.7,32.0,https://huggingface.co/microsoft/phi-1
    # phi-1.5,23.2,37.7,27.5,39.1,https://huggingface.co/microsoft/phi-1_5
    # "

    import numpy as np
    import pandas as pd

    df_crux = pd.read_csv("https://crux-eval.github.io/data.csv")
    df_crux = df_crux.replace("-", np.nan)

    # # Create a new DataFrame with the specified columns
    new_df = pd.DataFrame()
    new_df["model_name"] = df_crux["Model"]
    df_crux[["i@1", "i@5", "o@1", "o@5"]] = df_crux[["i@1", "i@5", "o@1", "o@5"]].astype(float)

    normalise("i@1", df_crux, new_df)
    normalise("i@5", df_crux, new_df)
    normalise("o@1", df_crux, new_df)
    normalise("o@5", df_crux, new_df)

    new_df["score"] = new_df[["i@1", "i@5", "o@1", "o@5"]].mean(axis=1)
    return new_df


@lru_cache(maxsize=1)
def fetch_tabby_leaderboard2():
    # prompt: Please do the same for the datasource: "https://leaderboard.tabbyml.com/tabby.yml"
    # It is of datatype: "yaml"
    # here is an example of the data:
    # "
    # DeepSeekCoder-V2-Lite:
    #   Baseline:
    #     Java: 36.65
    #     C#: 49.89
    #     Typescript: 30.84
    #     Python: 34.11
    #   BM25:
    #     Java: 44.18
    #     C#: 53.17
    #     Typescript: 32.21
    #     Python: 38.95
    #   Oracle:
    #     Java: 49.32
    #     C#: 58.31
    #     Typescript: 36.08
    #     Python: 44.02
    # DeepSeekCoder-6.7B:
    #   Baseline:
    #     Java: 37.87
    #     C#: 50.34
    #     Typescript: 32.48
    #     Python: 34.41
    #   BM25:
    #     Java: 44.23
    #     C#: 52.6
    #     Typescript: 34.03
    #     Python: 38.57
    #   Oracle:
    #     Java: 49.23
    #     C#: 58.2
    #     Typescript: 37.93
    #     Python: 44.39
    # "
    # I want you to create a dataframe with columns: "model_name" and "score"
    # I want you to add other columns for each score.  for example:
    # model_name = DeepSeekCoder-V2-Lite_baseline
    # java_score = 36.65
    # c#_score = 49.89
    # typescript_score = 30.84
    # python_score = 34.11
    # please then normalise the scores (so the scores are between 0 and 1 for each column)
    # and then create a new column: "score" which is the mean of all other scores
    # then order the rows by "score" highest to lowest

    import pandas as pd
    import yaml
    from urllib.request import urlopen

    try:
        with urlopen("https://leaderboard.tabbyml.com/tabby.yml") as file:
            data = yaml.safe_load(file)
    except Exception as e:
        print(f"An error occurred: {e}")
        data = {}  # Assign an empty dictionary in case of error

    rows = []
    for model_name, model_data in data.items():
        for metric, scores in model_data.items():
            row = {"model_name": f"{model_name}_{metric.lower()}"}
            for lang, score in scores.items():
                row[f"{lang.lower()}_score"] = score
            rows.append(row)

    df = pd.DataFrame(rows)

    # Normalize scores
    score_columns = [col for col in df.columns if col.endswith("_score")]
    for col in score_columns:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    # Calculate the mean score
    df["score"] = df[score_columns].mean(axis=1)

    return df


@lru_cache(maxsize=1)
def fetch_aider_leaderboard2():
    import pandas as pd
    import io
    import numpy as np

    df = pd.read_json("https://aider.chat/assets/js/search-data.json").T

    old_leaderboard = df[df["title"] == "Code editing leaderboard"]["content"]
    new_leaderboard = df[df["title"] == "Polyglot leaderboard"]["content"]

    new_leaderboard = new_leaderboard.to_list()[0]
    old_leaderboard = old_leaderboard.to_list()[0]

    new_leaderboard = new_leaderboard.split("intervention. ")[-1].split(" . ")
    old_leaderboard = old_leaderboard.split("intervention. ")[-1].split(" . ")

    new_leaderboard = "\n".join(new_leaderboard)
    old_leaderboard = "\n".join(old_leaderboard)

    old_buffer = io.StringIO(old_leaderboard)
    new_buffer = io.StringIO(new_leaderboard)

    old_leaderboard = pd.read_csv(old_buffer, sep="|")
    old_leaderboard = old_leaderboard.rename(columns=lambda x: x.strip())
    new_leaderboard = pd.read_csv(new_buffer, sep="|")
    new_leaderboard = new_leaderboard.rename(columns=lambda x: x.strip())

    old_rows = []
    for index, row in old_leaderboard.iterrows():
        try:
            model_name = f"{row['Model'].strip()}_{row['Edit format'].strip()}"
            percent_completed = float(row["Percent completed correctly"].replace("%", ""))
            percent_format = float(row["Percent using correct edit format"].replace("%", ""))
            score = (percent_completed + percent_format) / 200
            old_rows.append({"model_name": model_name, "score": score})
        except (ValueError, AttributeError, KeyError):
            pass

    old_df = pd.DataFrame(old_rows)

    new_rows = []
    for index, row in new_leaderboard.iterrows():
        try:
            model_name = f"{row['Model'].strip()}_{row['Edit format'].strip()}"
            percent_completed = float(row["Percent correct"].replace("%", ""))
            percent_format = float(row["Percent using correct edit format"].replace("%", ""))
            score = (percent_completed + percent_format) / 200
            new_rows.append({"model_name": model_name, "score": score})
        except (ValueError, AttributeError, KeyError):
            pass

    new_df = pd.DataFrame(new_rows)

    merged_df = pd.merge(old_df, new_df, on="model_name", how="outer", validate="many_to_many")

    # Calculate the mean of 'score' where both dataframes have values
    merged_df["score"] = merged_df.apply(
        lambda row: (
            np.mean([row["score_x"], row["score_y"]])
            if pd.notna(row["score_x"]) and pd.notna(row["score_y"])
            else row["score_x"] if pd.notna(row["score_x"]) else row["score_y"]
        ),
        axis=1,
    )

    # Drop the intermediate score columns
    merged_df = merged_df.drop(["score_x", "score_y"], axis=1, errors="ignore")
    return merged_df


def merge_dataframes(dfs):
    import pandas as pd

    merged_df = pd.DataFrame(columns=["model_name", "score"])
    for df in dfs:
        for index, row in df.iterrows():
            model_name = row["model_name"]
            row_score = row["score"]
            if model_name in merged_df["model_name"].values:
                merged_df.loc[merged_df["model_name"] == model_name, "score"] = (
                    merged_df.loc[merged_df["model_name"] == model_name, "score"].astype(float) + row_score
                ) / 2
            else:
                merged_df = pd.concat(
                    [merged_df, pd.DataFrame({"model_name": [model_name], "score": [row_score]})], ignore_index=True
                )
    return merged_df


def get_leaderboard_score(model_name: str) -> float:
    """Aggregates scores across all tests and leaderboards using normalized averaging."""

    merged_df = get_merged_leaderboard_df()

    scores = merged_df[merged_df["model_name"].str.contains(model_name)]
    if len(scores.index) == 0:
        return 0
    mean_score = float(scores["score"].mean())
    return mean_score
    # for source in sources:
    #     leaderboard_scores = []
    #     # Find all matching models in this leaderboard
    #     for leaderboard_model in source:
    #         if normalized in leaderboard_model:
    #             # Collect all scores from all tests for this model entry
    #             for test_scores in source[leaderboard_model].values():
    #                 leaderboard_scores.extend(test_scores)
    #
    #     if leaderboard_scores:
    #         # Calculate average for this leaderboard
    #         leaderboard_avg = sum(leaderboard_scores) / len(leaderboard_scores)
    #         leaderboard_averages.append(leaderboard_avg)
    #
    # if not leaderboard_averages:
    #     return 0
    #
    # # Calculate final score as average of leaderboard averages
    # final_score = sum(leaderboard_averages) / len(leaderboard_averages)
    # return final_score


@lru_cache(maxsize=1)
def get_merged_leaderboard_df():
    sources = [
        fetch_bigcodebench_leaderboard_easy(),
        fetch_bigcodebench_leaderboard_hard(),
        fetch_evalplus_leaderboard2(),
        fetch_crux_leaderboard2(),
        fetch_tabby_leaderboard2(),
        fetch_aider_leaderboard2(),
    ]
    merged_df = merge_dataframes(sources)
    merged_df = merged_df.sort_values(by="score", ascending=False)
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
