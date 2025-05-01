import io
import json  # Import json
import logging
import re  # Import regex module
from functools import lru_cache
from typing import Any, Dict, List, Union, Callable

import coloredlogs
import numpy as np
import pandas as pd
import requests
import yaml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.consts import LOG_LEVEL

# Configure logging
logger = logging.getLogger(__name__)
coloredlogs.install(level=LOG_LEVEL, logger=logger)

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
    Handles non-numeric data and lack of variation.
    """
    # if column not in target_df.columns:
    #     logger.warning(f"Column '{column}' not found in target DataFrame for normalization.")
    #     target_df[column] = np.nan
    # return
    if column not in source_df.columns:
        logger.warning(f"Column '{column}' not found in source DataFrame for normalization.")
        source_df[column] = np.nan
        # return

    # Ensure the column is numeric, coercing errors to NaN
    source_numeric = pd.to_numeric(source_df[column], errors="coerce")

    min_val = source_numeric.min()
    max_val = source_numeric.max()

    if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
        # Handle cases with no variation or all NaNs
        # Assign NaN where normalization isn't possible/meaningful
        target_df[column] = min_val
        # logger.warning(f"Could not normalize column '{column}' due to missing values or no variation.")
    else:
        # Apply normalization only to numeric values in the target DataFrame
        target_numeric = pd.to_numeric(target_df[column], errors="coerce")
        target_df[column] = (target_numeric - min_val) / (max_val - min_val)


@lru_cache(maxsize=1)
def fetch_bigcodebench_leaderboard_easy() -> pd.DataFrame:
    """
    Fetches and processes the 'easy' BigCodeBench leaderboard data.
    """
    url = "https://bigcode-bench.github.io/results.json"
    try:
        response = SESSION.get(url, timeout=15)
        response.raise_for_status()
        # Read from response text
        df1 = pd.read_json(io.StringIO(response.text))
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch BigCodeBench easy data from {url}: {e}")
        return pd.DataFrame(columns=["model_name", "score"])
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from BigCodeBench easy data {url}: {e}")
        return pd.DataFrame(columns=["model_name", "score"])
    except Exception as e:
        logger.error(f"An unexpected error occurred processing BigCodeBench easy data: {e}")
        return pd.DataFrame(columns=["model_name", "score"])

    new_rows = []
    for col in df1.columns:
        try:
            pass_at_1_data = df1.get(col, {}).get("pass@1")
            if isinstance(pass_at_1_data, dict):
                instruct_score = pass_at_1_data.get("instruct")
                complete_score = pass_at_1_data.get("complete")

                if instruct_score is not None:
                    new_rows.append({"model_name": f"{col}_instruct", "score": instruct_score})
                if complete_score is not None:
                    new_rows.append({"model_name": f"{col}_complete", "score": complete_score})
            elif pass_at_1_data is not None:
                logger.warning("Unexpected format for 'pass@1' in column '%s' (Easy), metrics: %s", col, pass_at_1_data)

        except Exception as e:
            logger.warning("Error processing data in column '%s' (Easy): %s", col, e)

    if not new_rows:
        logger.warning("No valid rows generated for BigCodeBench easy leaderboard.")
        return pd.DataFrame(columns=["model_name", "score"])

    new_df = pd.DataFrame(new_rows)
    if not new_df["score"].empty and new_df["score"].notna().any():
        normalize_column(new_df, new_df, "score")
    else:
        logger.warning("Score column is empty or all NaN for BigCodeBench easy, skipping normalization.")
        new_df["score"] = np.nan

    return new_df[["model_name", "score"]]


@lru_cache(maxsize=1)
def fetch_bigcodebench_leaderboard_hard() -> pd.DataFrame:
    """
    Fetches and processes the 'hard' BigCodeBench leaderboard data.
    """
    url = "https://bigcode-bench.github.io/results-hard.json"
    try:
        response = SESSION.get(url, timeout=15)
        response.raise_for_status()
        df1 = pd.read_json(io.StringIO(response.text))
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch BigCodeBench hard data from {url}: {e}")
        return pd.DataFrame(columns=["model_name", "score"])
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from BigCodeBench hard data {url}: {e}")
        return pd.DataFrame(columns=["model_name", "score"])
    except Exception as e:
        logger.error(f"An unexpected error occurred processing BigCodeBench hard data: {e}")
        return pd.DataFrame(columns=["model_name", "score"])

    new_rows = []
    for col in df1.columns:
        try:
            pass_at_1_data = df1.get(col, {}).get("pass@1")
            if isinstance(pass_at_1_data, dict):
                instruct_score = pass_at_1_data.get("instruct")
                complete_score = pass_at_1_data.get("complete")

                if instruct_score is not None:
                    new_rows.append({"model_name": f"{col}_instruct", "score": instruct_score})
                if complete_score is not None:
                    new_rows.append({"model_name": f"{col}_complete", "score": complete_score})
            elif pass_at_1_data is not None:
                logger.warning("Unexpected format for 'pass@1' in column '%s' (Hard), metrics: %s", col, pass_at_1_data)

        except Exception as e:
            logger.warning("Error processing data in column '%s' (Hard): %s", col, e)

    if not new_rows:
        logger.warning("No valid rows generated for BigCodeBench hard leaderboard.")
        return pd.DataFrame(columns=["model_name", "score"])

    new_df = pd.DataFrame(new_rows)
    if not new_df["score"].empty and new_df["score"].notna().any():
        normalize_column(new_df, new_df, "score")
    else:
        logger.warning("Score column is empty or all NaN for BigCodeBench hard, skipping normalization.")
        new_df["score"] = np.nan

    return new_df[["model_name", "score"]]


@lru_cache(maxsize=1)
def fetch_evalplus_leaderboard2() -> pd.DataFrame:
    """
    Fetches EvalPlus leaderboard data and computes the mean of various scores.
    Handles missing scores correctly during mean calculation.
    """
    url = "https://evalplus.github.io/results.json"
    try:
        response = SESSION.get(url, timeout=15)
        response.raise_for_status()
        df = pd.read_json(io.StringIO(response.text))
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch EvalPlus data from {url}: {e}")
        return pd.DataFrame(columns=["model_name", "score"])
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON from EvalPlus data {url}: {e}")
        return pd.DataFrame(columns=["model_name", "score"])
    except Exception as e:
        logger.error(f"An unexpected error occurred processing EvalPlus data: {e}")
        return pd.DataFrame(columns=["model_name", "score"])

    new_rows = []
    score_cols = ["humaneval_score", "humaneval+_score", "mbpp_score", "mbpp+_score"]
    original_score_keys = ["humaneval", "humaneval+", "mbpp", "mbpp+"]  # Keys in the source JSON

    for model_name, model_data in df.items():
        try:
            pass_at_1 = model_data.get("pass@1")
            row_data = {"model_name": model_name}
            found_score = False
            if isinstance(pass_at_1, dict):
                for df_col, json_key in zip(score_cols, original_score_keys):
                    score_val = pass_at_1.get(json_key)
                    # Ensure score is numeric, otherwise use NaN
                    try:
                        numeric_score = float(score_val) if score_val is not None else np.nan
                    except (ValueError, TypeError):
                        logger.warning(
                            f"Non-numeric score '{score_val}' found for {json_key} in model {model_name}. "
                            f"Setting to NaN."
                        )
                        numeric_score = np.nan
                    row_data[df_col] = numeric_score
                    if pd.notna(numeric_score):
                        found_score = True
                if found_score:
                    new_rows.append(row_data)
                else:
                    logger.info(
                        f"Skipping model {model_name} in EvalPlus as no valid scores found in pass@1: {pass_at_1}"
                    )

            elif pass_at_1 is not None:
                logger.warning(f"Unexpected format for pass@1 for model {model_name} in EvalPlus: {pass_at_1}")
        except Exception as e:
            logger.warning("Error processing model %s in EvalPlus: %s", model_name, e, exc_info=True)

    if not new_rows:
        logger.warning("No valid rows generated for EvalPlus leaderboard.")
        return pd.DataFrame(columns=["model_name", "score"])

    new_df = pd.DataFrame(new_rows)

    # Normalize each score column individually if it has valid data
    valid_cols_for_norm = []
    for col in score_cols:
        if col in new_df.columns and new_df[col].notna().any():
            normalize_column(new_df, new_df, col)
            valid_cols_for_norm.append(col)
        else:
            # Ensure column exists even if all NaN, for the mean calculation later
            if col not in new_df.columns:
                new_df[col] = np.nan
            logger.info(f"Skipping normalization for EvalPlus column '{col}' due to only NaN values.")

    # Calculate the mean score across the normalized columns (mean ignores NaNs by default)
    if valid_cols_for_norm:
        logger.info(f"Calculating mean score for EvalPlus based on columns: {valid_cols_for_norm}")
        new_df["score"] = new_df[valid_cols_for_norm].mean(axis=1)
    else:
        logger.warning("No columns were successfully normalized in EvalPlus. Assigning NaN to 'score'.")
        new_df["score"] = np.nan

    return new_df[["model_name", "score"]]


@lru_cache(maxsize=1)
def fetch_crux_leaderboard2() -> pd.DataFrame:
    """
    Fetches Crux leaderboard data from CSV and processes it.
    """
    url = "https://crux-eval.github.io/data.csv"
    try:
        df_crux = pd.read_csv(url)
    except Exception as e:
        logger.error(f"Failed to fetch or parse Crux data from {url}: {e}")
        return pd.DataFrame(columns=["model_name", "score"])

    df_crux = df_crux.replace("-", np.nan)
    score_cols_crux = ["i@1", "i@5", "o@1", "o@5"]

    for col in score_cols_crux:
        if col in df_crux.columns:
            df_crux[col] = pd.to_numeric(df_crux[col], errors="coerce")
        else:
            logger.warning(f"Column '{col}' not found in Crux data.")

    if "Model" not in df_crux.columns:
        logger.error("Crux data is missing the 'Model' column.")
        return pd.DataFrame(columns=["model_name", "score"])

    # new_df = pd.DataFrame()
    # new_df["model_name"] = df_crux["Model"]

    valid_cols_for_norm_crux = []
    for col in score_cols_crux:
        if col in df_crux.columns and df_crux[col].notna().any():
            normalize_column(df_crux, df_crux, col)  # Pass df_crux as source
            valid_cols_for_norm_crux.append(col)
        else:
            # Ensure column exists in new_df for mean calculation
            df_crux[col] = np.nan
            logger.info(f"Skipping normalization for Crux column '{col}' due to missing or only NaN values.")

    if valid_cols_for_norm_crux:
        logger.info(f"Calculating mean score for Crux based on columns: {valid_cols_for_norm_crux}")
        df_crux["score"] = df_crux[valid_cols_for_norm_crux].mean(axis=1)
    else:
        logger.warning("No valid columns found for score calculation in Crux.")
        df_crux["score"] = np.nan

    df_crux = df_crux.rename(columns={"Model": "model_name"})
    return df_crux[["model_name", "score"]]


@lru_cache(maxsize=1)
def fetch_tabby_leaderboard2() -> pd.DataFrame:
    """
    Fetches Tabby leaderboard data from YAML, normalizes the scores, and calculates the mean score.
    """
    url = "https://leaderboard.tabbyml.com/tabby.yml"
    data = {}
    try:
        response = SESSION.get(url, timeout=10)
        response.raise_for_status()
        data = yaml.safe_load(response.text)
    except requests.exceptions.RequestException as e:
        logger.error("An error occurred while fetching Tabby data: %s", e)
    except yaml.YAMLError as e:
        logger.error("An error occurred while parsing Tabby YAML data: %s", e)
    except Exception as e:
        logger.error("An unexpected error occurred processing Tabby data: %s", e)

    if not data or not isinstance(data, dict):
        logger.error("Fetched Tabby data is empty or not in expected dictionary format.")
        return pd.DataFrame(columns=["model_name", "score"])

    rows = []
    for model_name, model_data in data.items():
        if not isinstance(model_data, dict):
            logger.warning(f"Skipping model '{model_name}' in Tabby data: invalid format '{type(model_data)}'.")
            continue
        for metric, scores in model_data.items():
            if not isinstance(scores, dict):
                logger.warning(
                    f"Skipping metric '{metric}' for model '{model_name}' in Tabby data: "
                    f"invalid scores format '{type(scores)}'."
                )
                continue

            row: Dict[str, Union[str, float, int]] = {"model_name": f"{model_name}_{metric.lower()}"}
            for lang, score in scores.items():
                if isinstance(lang, str):
                    # Ensure score is numeric
                    try:
                        numeric_score = float(score) if score is not None else np.nan
                    except (ValueError, TypeError):
                        logger.warning(
                            f"Non-numeric score '{score}' for lang '{lang}' in Tabby model '{model_name}', "
                            f"metric '{metric}'. Setting to NaN."
                        )
                        numeric_score = np.nan
                    row[f"{lang.lower()}_score"] = numeric_score
                else:
                    logger.warning(
                        f"Skipping language '{lang}' for metric '{metric}', "
                        f"model '{model_name}' in Tabby data: non-string key."
                    )
            rows.append(row)

    if not rows:
        logger.warning("No valid rows generated for Tabby leaderboard.")
        return pd.DataFrame(columns=["model_name", "score"])

    df = pd.DataFrame(rows)
    score_columns = [col for col in df.columns if col.endswith("_score")]

    valid_cols_for_norm_tabby = []
    for col in score_columns:
        if col in df.columns and df[col].notna().any():
            normalize_column(df, df, col)  # Pass copy as source_df
            valid_cols_for_norm_tabby.append(col)
        else:
            # Ensure column exists for mean calculation
            if col not in df.columns:
                df[col] = np.nan
            logger.info(f"Skipping normalization for Tabby column '{col}' due to only NaN values.")

    if valid_cols_for_norm_tabby:
        logger.info(f"Calculating mean score for Tabby based on columns: {valid_cols_for_norm_tabby}")
        df["score"] = df[valid_cols_for_norm_tabby].mean(axis=1)
    else:
        logger.warning("No valid score columns found for score calculation in Tabby.")
        df["score"] = np.nan

    df = df.sort_values(by="score", ascending=False)
    df.reset_index(drop=True, inplace=True)
    return df[["model_name", "score"]]


def process_leaderboard(
    df: pd.DataFrame, model_col: str, edit_format_col: str, percent_col: str, format_col: str
) -> List[Dict[str, Any]]:
    """Processes the raw DataFrame from markdown into the desired score format."""
    rows: List[Dict[str, Union[str, float, int]]] = []
    required_cols = [model_col, edit_format_col, percent_col, format_col]
    if not all(col in df.columns for col in required_cols):
        logger.error(
            f"Missing required columns in input DataFrame for process_leaderboard. Required: {required_cols}, "
            f"Found: {df.columns.tolist()}"
        )
        return rows

    for _, row in df.iterrows():
        try:
            model_val = str(row[model_col]) if pd.notna(row[model_col]) else ""
            edit_format_val = str(row[edit_format_col]) if pd.notna(row[edit_format_col]) else ""
            percent_val = str(row[percent_col]) if pd.notna(row[percent_col]) else ""
            format_val = str(row[format_col]) if pd.notna(row[format_col]) else ""

            model_name = f"{model_val.strip()}_{edit_format_val.strip()}"

            try:
                percent_completed = float(percent_val.replace("%", "").strip())
            except (ValueError, AttributeError):  # Catch AttributeError if strip fails on non-string
                # logger.warning(f"Could not convert '{percent_val}' to float for completed % in row: {row.to_dict()}")
                percent_completed = np.nan

            try:
                percent_format = float(format_val.replace("%", "").strip())
            except (ValueError, AttributeError):
                # logger.warning(f"Could not convert '{format_val}' to float for format % in row: {row.to_dict()}")
                percent_format = np.nan

            if pd.notna(percent_completed) and pd.notna(percent_format):
                score = (percent_completed + percent_format) / 200.0
                rows.append({"model_name": model_name, "score": score})
            else:
                rows.append({"model_name": model_name, "score": np.nan})

        except (AttributeError, KeyError) as e:
            logger.warning(f"Error processing row in process_leaderboard: {row.to_dict()}. Error: {e}")
            continue
    return rows


def get_leaderboard_df(
    leaderboard_text: str, process_func: Callable, header_pattern: str, line_pattern: str
) -> pd.DataFrame:
    """
    Process leaderboard markdown text into DataFrame using more robust parsing.
    """
    if not leaderboard_text or not isinstance(leaderboard_text, str):
        logger.warning("Invalid or empty leaderboard text provided.")
        return pd.DataFrame(columns=["model_name", "score"])

    try:
        # Find the header line more reliably, allowing optional | at start/end
        header_match = re.search(header_pattern, leaderboard_text, re.MULTILINE | re.IGNORECASE)
        if not header_match:
            logger.warning("Could not find header line in Aider leaderboard data")
            return pd.DataFrame(columns=["model_name", "score"])

        header_line = header_match.group(0).strip()
        table_start_index = header_match.end()

        separator_line_end_index = table_start_index

        # Extract data rows starting after the separator line
        # Match lines starting with '|' until a line that doesn't start with '|' or end of string
        data_lines_text = ""
        current_pos = separator_line_end_index
        while current_pos < len(leaderboard_text):
            line_match = re.match(line_pattern, leaderboard_text[current_pos:])
            if line_match:
                data_lines_text += f"{line_match.group(0)}\n"
                current_pos += line_match.end()
            else:
                break  # Stop when a non-table row is encountered

        if not data_lines_text:
            logger.warning("Could not extract table content rows from Aider leaderboard.")
            return pd.DataFrame(columns=["model_name", "score"])

        # Combine header and data lines for CSV parsing
        table_csv_text = header_line + "\n" + data_lines_text

        buffer = io.StringIO(table_csv_text)

        df = pd.read_csv(buffer, sep="|", skipinitialspace=True, index_col=False, on_bad_lines="warn")

        # Clean up DataFrame
        if not df.empty:
            # Drop potentially empty first/last columns created by leading/trailing '|'
            df = df.iloc[:, 1:-1]
            df.columns = df.columns.str.strip()
            df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
        else:
            logger.warning("pd.read_csv resulted in an empty DataFrame for Aider leaderboard.")
            return pd.DataFrame(columns=["model_name", "score"])

        # Find column names robustly
        model_col_name = next((col for col in df.columns if "model" in col.lower()), None)
        edit_format_col_name = next(
            (col for col in df.columns if "edit format" in col.lower() and "correct" not in col.lower()), None
        )
        percent_completed_col_name = next(
            (col for col in df.columns if "percent completed" in col.lower() or "percent correct" in col.lower()), None
        )
        percent_format_col_name = next(
            (
                col
                for col in df.columns
                if "percent using correct" in col.lower() or "correct edit format" in col.lower()
            ),
            None,
        )

        required_cols_map = {
            "Model": model_col_name,
            "Edit format": edit_format_col_name,
            "Percent completed correctly": percent_completed_col_name,
            "Percent using correct edit format": percent_format_col_name,
        }

        if not all(required_cols_map.values()):
            missing = [k for k, v in required_cols_map.items() if v is None]
            logger.error(
                f"Could not find all required columns in parsed Aider table. "
                f"Missing: {missing}. Found columns: {df.columns.tolist()}"
            )
            return pd.DataFrame(columns=["model_name", "score"])

        # Process the dataframe
        result_rows = process_func(
            df,
            required_cols_map["Model"],
            required_cols_map["Edit format"],
            required_cols_map["Percent completed correctly"],
            required_cols_map["Percent using correct edit format"],
        )

        if not result_rows:
            logger.warning("Aider processing function returned no rows.")
            return pd.DataFrame(columns=["model_name", "score"])

        return pd.DataFrame(result_rows)

    except Exception as e:
        logger.error("Error processing Aider leaderboard markdown: %s", e, exc_info=True)
        return pd.DataFrame(columns=["model_name", "score"])


@lru_cache(maxsize=1)
def fetch_aider_leaderboard2() -> pd.DataFrame:
    """
    Fetches Aider leaderboard data from the internet, processes both old and new formats,
    and merges them.
    """
    data = None
    try:
        url = "https://aider.chat/assets/js/search-data.json"
        logger.info(f"Fetching Aider leaderboard data from {url}")
        response = SESSION.get(url, timeout=20)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching Aider leaderboard from {url}: {e}")
        return pd.DataFrame(columns=["model_name", "score"])
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {url}: {e}")
        return pd.DataFrame(columns=["model_name", "score"])
    except Exception as e:
        logger.error(f"An unexpected error occurred fetching Aider data: {e}", exc_info=True)
        return pd.DataFrame(columns=["model_name", "score"])

    if not data or not isinstance(data, dict):
        logger.error("Fetched Aider data is empty or not in the expected format.")
        return pd.DataFrame(columns=["model_name", "score"])

    old_leaderboard_content = None
    new_leaderboard_content = None
    for key, value in data.items():
        if isinstance(value, dict):
            title = value.get("title")
            if title == "Code editing leaderboard":
                old_leaderboard_content = value.get("content")
            elif title == "Aider polyglot coding leaderboard":
                new_leaderboard_content = value.get("content")
        if old_leaderboard_content and new_leaderboard_content:
            break

    if not old_leaderboard_content:
        logger.warning("Could not find 'Code editing leaderboard' content in Aider JSON.")
    if not new_leaderboard_content:
        logger.warning("Could not find 'Aider polyglot coding leaderboard' content in Aider JSON.")

    old_processed = pd.DataFrame(columns=["model_name", "score"])
    if old_leaderboard_content:
        logger.info("Processing old Aider leaderboard...")
        header_pattern = (
            r" \| Model \| Percent completed correctly \| "
            r"Percent using correct edit format \| Command \| Edit format \| \. "
        )
        line_pattern = r".*?\| \. "
        old_processed = get_leaderboard_df(old_leaderboard_content, process_leaderboard, header_pattern, line_pattern)
        if old_processed.empty:
            logger.warning("Old Aider leaderboard processing resulted in an empty DataFrame.")

    new_processed = pd.DataFrame(columns=["model_name", "score"])
    if new_leaderboard_content:
        logger.info("Processing new Aider leaderboard...")
        header_pattern = (
            r" \| Model \| Percent correct \| Cost \| Command \| Correct edit format \| Edit Format \| \. \| ▶ "
        )
        # line_pattern = r"\|[ \.▶].*?\| \. \| "
        line_pattern = r".*? ▶ "
        new_processed = get_leaderboard_df(new_leaderboard_content, process_leaderboard, header_pattern, line_pattern)
        if new_processed.empty:
            logger.warning("New Aider leaderboard processing resulted in an empty DataFrame.")

    if old_processed.empty and new_processed.empty:
        logger.warning("Both Aider leaderboards resulted in empty DataFrames.")
        return pd.DataFrame(columns=["model_name", "score"])
    elif old_processed.empty:
        logger.info("Using only new Aider leaderboard data.")
        return new_processed
    elif new_processed.empty:
        logger.info("Using only old Aider leaderboard data.")
        return old_processed

    logger.info("Merging old and new Aider leaderboards.")
    try:
        old_processed["score"] = pd.to_numeric(old_processed["score"], errors="coerce")
        new_processed["score"] = pd.to_numeric(new_processed["score"], errors="coerce")

        merged_df = pd.merge(old_processed, new_processed, on="model_name", how="outer", suffixes=("_old", "_new"))

        merged_df["score"] = np.nanmean(merged_df[["score_old", "score_new"]], axis=1)
        merged_df = merged_df.drop(["score_old", "score_new"], axis=1, errors="ignore")
        merged_df = merged_df.dropna(subset=["score"])

        if merged_df.empty:
            logger.warning("Merged Aider DataFrame is empty after dropping NaN scores.")

        return merged_df

    except Exception as e:
        logger.error(f"Error merging Aider leaderboards: {e}", exc_info=True)
        if not new_processed.empty:
            return new_processed
        if not old_processed.empty:
            return old_processed
        return pd.DataFrame(columns=["model_name", "score"])


def merge_dataframes(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Merges multiple dataframes of leaderboard scores into one, averaging scores for duplicate model names.
    """
    if not dfs:
        return pd.DataFrame(columns=["model_name", "score"])

    all_dfs = [
        df
        for df in dfs
        if isinstance(df, pd.DataFrame) and not df.empty and "model_name" in df.columns and "score" in df.columns
    ]
    if not all_dfs:
        logger.warning("No valid dataframes provided to merge_dataframes.")
        return pd.DataFrame(columns=["model_name", "score"])

    merged_df = pd.concat(all_dfs, ignore_index=True)
    merged_df["score"] = pd.to_numeric(merged_df["score"], errors="coerce")

    # Calculate mean, implicitly skipping NaNs
    final_df = merged_df.groupby("model_name", as_index=False)["score"].mean()

    # Drop rows where the final averaged score is NaN
    final_df = final_df.dropna(subset=["score"])

    logger.info(f"Merging resulted in {len(final_df)} unique models with scores.")
    return final_df


@lru_cache(maxsize=1)
def get_merged_leaderboard_df() -> pd.DataFrame:
    """
    Merges all leaderboard data sources fetched from the internet into a single sorted dataframe.
    """
    sources = []
    # Use a dictionary for clarity and easier modification
    fetch_functions = {
        "Aider": fetch_aider_leaderboard2,
        "BigCodeBench Easy": fetch_bigcodebench_leaderboard_easy,
        "BigCodeBench Hard": fetch_bigcodebench_leaderboard_hard,
        "EvalPlus": fetch_evalplus_leaderboard2,
        "Crux": fetch_crux_leaderboard2,
        "Tabby": fetch_tabby_leaderboard2,
    }

    for name, func in fetch_functions.items():
        try:
            logger.info(f"Fetching {name} data...")
            df = func()
            if isinstance(df, pd.DataFrame) and not df.empty:
                if "model_name" in df.columns and "score" in df.columns:
                    df_cleaned = df[["model_name", "score"]].copy()
                    df_cleaned["score"] = pd.to_numeric(df_cleaned["score"], errors="coerce")
                    df_cleaned = df_cleaned.dropna(subset=["score"])
                    if not df_cleaned.empty:
                        sources.append(df_cleaned)
                        logger.info(f"Successfully processed {name} data ({len(df_cleaned)} rows).")
                    else:
                        logger.warning(f"{name} data resulted in empty DataFrame after cleaning NaN scores.")
                else:
                    logger.warning(
                        f"{name} data is missing required columns ('model_name', 'score'). Found: {df.columns.tolist()}"
                    )
            elif isinstance(df, pd.DataFrame) and df.empty:
                logger.warning(f"{name} data source returned an empty DataFrame.")
            else:
                logger.warning(f"{name} function did not return a DataFrame.")
        except Exception as e:
            logger.error(f"Error processing data source {name}: {e}", exc_info=True)

    if not sources:
        logger.warning("No data sources could be successfully processed. Returning empty DataFrame.")
        return pd.DataFrame(columns=["model_name", "score"])

    logger.info(f"Merging data from {len(sources)} sources.")
    merged_df = merge_dataframes(sources)

    if merged_df.empty:
        logger.warning("Merging resulted in an empty DataFrame.")
        return merged_df

    merged_df.sort_values(by="score", ascending=False, inplace=True)
    merged_df.reset_index(drop=True, inplace=True)
    logger.info(f"Final merged DataFrame has {len(merged_df)} rows.")
    return merged_df


def get_leaderboard_score(model_name: str) -> float:
    """
    Aggregates scores across all tests and leaderboards using normalized averaging.
    Fetches data online.
    """
    merged_df = get_merged_leaderboard_df()  # Fetches online data

    if merged_df.empty or "model_name" not in merged_df.columns or "score" not in merged_df.columns:
        logger.warning(f"Merged leaderboard is empty or invalid when searching for {model_name}.")
        return 0.0

    # Use case-insensitive matching and handle potential NaNs in model_name column
    scores = merged_df[
        merged_df["model_name"].str.contains(model_name, case=False, na=False, regex=False)
    ]  # Use regex=False for literal matching

    if scores.empty:
        # logger.info(f"No scores found containing '{model_name}' (case-insensitive).")
        return 0.0

    valid_scores = pd.to_numeric(scores["score"], errors="coerce").dropna()

    if valid_scores.empty:
        logger.warning(f"Found entries for '{model_name}', but scores were non-numeric or NaN.")
        return 0.0

    mean_score = float(valid_scores.mean())
    # logger.info(f"Calculated mean score for '{model_name}': {mean_score:.4f} from {len(valid_scores)} entries.")
    return mean_score


if __name__ == "__main__":
    # Now runs fetching data online
    models = [
        "Magicoder-S-DS-6.7B",
        "StarCoder2-15B-Instruct-v0.1",
        "OpenCoder-8B-Instruct",
        "phi-1",
        "DeepSeekCoder-V2-Lite",
        "claude-3-5-sonnet-20241022",  # Example from aider leaderboard
        "phi-1.5",
        "KwaiCoder-23B-A4B-v1",
        "Qwen-QWQ",  # This is likely a specific variant, check exact names
        "gpt-4",  # Broad term, might match multiple specific versions
        "gpt-4o",  # Broad term
        "gpt-3.5",  # Broad term
        "o1",  # From aider leaderboard
        "gemini-exp-1206",  # From aider leaderboard
        "DeepSeek Coder V2 0724",  # From aider leaderboard
        "gemma",  # Broad term
        "llama",  # Broad term
    ]

    model_scores = []
    print("Fetching and calculating scores...")
    for model in models:
        logger.info(f"\n--- Getting score for: {model} ---")
        score = get_leaderboard_score(model)
        if score > 0.0:
            model_scores.append((model, score))
            logger.info(f"Found score for {model}: {score:.4f}")
        else:
            logger.info("No valid data found for %s", model)

    # Sort models by score in descending order
    model_scores.sort(key=lambda x: x[1], reverse=True)

    # Print ranked results
    print("\n--- Rankings ---")
    if model_scores:
        for rank, (model, score) in enumerate(model_scores, 1):
            print(f"Rank {rank}: {model} - Aggregated Score: {score:.4f}")
    else:
        print("No models found with valid scores.")

    # Optional: Print the full merged DataFrame for inspection
    print("\n--- Full Merged Leaderboard (Top 20) ---")
    # Fetch again (or reuse if not performance critical)
    final_df = get_merged_leaderboard_df()
    if not final_df.empty:
        print(final_df.head(20).to_string())
    else:
        print("Merged DataFrame is empty.")
