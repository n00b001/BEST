import io
import logging
from copy import deepcopy
from functools import lru_cache
from typing import Any, Dict, List
from urllib.request import urlopen
import re # Import regex module

import numpy as np
import pandas as pd
import requests
import yaml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- (Keep existing logging and session configuration) ---
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

# --- (Keep existing normalize_column, fetch_bigcodebench*, fetch_evalplus*, fetch_crux*, fetch_tabby*) ---
def normalize_column(source_df: pd.DataFrame, target_df: pd.DataFrame, column: str) -> None:
    """
    Normalizes the values of a column in source_df and assigns them to target_df.
    """
    # Ensure the column is numeric, coercing errors to NaN
    source_numeric = pd.to_numeric(source_df[column], errors='coerce')
    min_val = source_numeric.min()
    max_val = source_numeric.max()

    if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
        # Handle cases with no variation or all NaNs - assign 0 or keep as NaN/original
        # Assigning 0 or 1 might be appropriate depending on context. Let's assign NaN where normalization isn't possible.
        target_df[column] = np.nan
        logger.warning(f"Could not normalize column '{column}' due to missing values or no variation.")
    else:
        target_df[column] = (source_numeric - min_val) / (max_val - min_val)

# --- fetch_bigcodebench_leaderboard_easy (keep as is) ---
@lru_cache(maxsize=1)
def fetch_bigcodebench_leaderboard_easy() -> pd.DataFrame:
    """
    Fetches and processes the 'easy' BigCodeBench leaderboard data.
    """
    url = "https://bigcode-bench.github.io/results.json"
    try:
        df1 = pd.read_json(url)
    except Exception as e:
        logger.error(f"Failed to fetch or parse BigCodeBench easy data from {url}: {e}")
        return pd.DataFrame(columns=["model_name", "score"])

    new_rows = []
    for col in df1.columns:
        try:
            # Check if 'pass@1' exists and is a dictionary
            pass_at_1_data = df1.get(col, {}).get("pass@1")
            if isinstance(pass_at_1_data, dict):
                instruct_score = pass_at_1_data.get("instruct")
                complete_score = pass_at_1_data.get("complete")

                if instruct_score is not None:
                    new_rows.append({"model_name": f"{col}_instruct", "score": instruct_score})
                if complete_score is not None:
                    new_rows.append({"model_name": f"{col}_complete", "score": complete_score})
            elif pass_at_1_data is not None: # Log if pass@1 exists but isn't a dict
                 logger.warning("Unexpected format for 'pass@1' in column '%s', metrics: %s", col, pass_at_1_data)

        except Exception as e: # Catch broader exceptions during processing
            logger.warning("Error processing data in column '%s': %s", col, e)

    if not new_rows:
        logger.warning("No valid rows generated for BigCodeBench easy leaderboard.")
        return pd.DataFrame(columns=["model_name", "score"])

    new_df = pd.DataFrame(new_rows)
    # Add check for empty score column before normalization
    if not new_df["score"].empty and new_df["score"].notna().any():
         normalize_column(new_df, new_df, "score")
    else:
        logger.warning("Score column is empty or all NaN for BigCodeBench easy, skipping normalization.")
        new_df["score"] = np.nan # Ensure score column exists even if empty/NaN

    return new_df

# --- fetch_bigcodebench_leaderboard_hard (keep as is, but add similar error handling as easy) ---
@lru_cache(maxsize=1)
def fetch_bigcodebench_leaderboard_hard() -> pd.DataFrame:
    """
    Fetches and processes the 'hard' BigCodeBench leaderboard data.
    """
    url = "https://bigcode-bench.github.io/results-hard.json"
    try:
        df1 = pd.read_json(url)
    except Exception as e:
        logger.error(f"Failed to fetch or parse BigCodeBench hard data from {url}: {e}")
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
                 logger.warning("Unexpected format for 'pass@1' in column '%s', metrics: %s", col, pass_at_1_data)

        except Exception as e:
            logger.warning("Error processing data in column '%s': %s", col, e)


    if not new_rows:
        logger.warning("No valid rows generated for BigCodeBench hard leaderboard.")
        return pd.DataFrame(columns=["model_name", "score"])

    new_df = pd.DataFrame(new_rows)
    if not new_df["score"].empty and new_df["score"].notna().any():
        normalize_column(new_df, new_df, "score")
    else:
        logger.warning("Score column is empty or all NaN for BigCodeBench hard, skipping normalization.")
        new_df["score"] = np.nan

    return new_df


# --- fetch_evalplus_leaderboard2 (keep as is, add check before normalization) ---
@lru_cache(maxsize=1)
def fetch_evalplus_leaderboard2() -> pd.DataFrame:
    """
    Fetches EvalPlus leaderboard data and computes the mean of various scores.
    """
    url = "https://evalplus.github.io/results.json"
    try:
        df = pd.read_json(url)
    except Exception as e:
        logger.error(f"Failed to fetch or parse EvalPlus data from {url}: {e}")
        return pd.DataFrame(columns=["model_name", "score"])

    new_rows = []
    for model_name, model_data in df.items():
        try:
            pass_at_1 = model_data.get("pass@1") # Use .get for safer access
            if isinstance(pass_at_1, dict):
                 # Check for existence of keys before accessing
                 humaneval = pass_at_1.get("humaneval")
                 humaneval_plus = pass_at_1.get("humaneval+")
                 mbpp = pass_at_1.get("mbpp")
                 mbpp_plus = pass_at_1.get("mbpp+")

                 # Only add row if all scores are present (or handle missing ones as needed)
                 if all(v is not None for v in [humaneval, humaneval_plus, mbpp, mbpp_plus]):
                     new_rows.append(
                         {
                             "model_name": model_name,
                             "humaneval_score": humaneval,
                             "humaneval+_score": humaneval_plus,
                             "mbpp_score": mbpp,
                             "mbpp+_score": mbpp_plus,
                         }
                     )
                 else:
                     logger.warning(f"Skipping model {model_name} in EvalPlus due to missing score(s) in pass@1: {pass_at_1}")
            elif pass_at_1 is not None:
                logger.warning(f"Unexpected format for pass@1 for model {model_name} in EvalPlus: {pass_at_1}")
        except Exception as e: # Catch broader exceptions
            logger.warning("Error processing model %s in EvalPlus: %s", model_name, e)

    if not new_rows:
         logger.warning("No valid rows generated for EvalPlus leaderboard.")
         return pd.DataFrame(columns=["model_name", "score"])

    new_df = pd.DataFrame(new_rows)
    score_cols = ["humaneval_score", "humaneval+_score", "mbpp_score", "mbpp+_score"]

    # Normalize only if columns exist and have valid data
    valid_cols_for_norm = [col for col in score_cols if col in new_df.columns and new_df[col].notna().any()]
    if not valid_cols_for_norm:
         logger.warning("No valid columns found for normalization in EvalPlus.")
         new_df["score"] = np.nan # Assign NaN score if normalization isn't possible
         return new_df[['model_name', 'score']] # Return only model_name and score

    for col in valid_cols_for_norm:
         normalize_column(new_df, new_df, col)

    # Calculate mean score only on successfully normalized columns
    new_df["score"] = new_df[valid_cols_for_norm].mean(axis=1)
    return new_df[['model_name', 'score']] # Keep only relevant columns


# --- fetch_crux_leaderboard2 (keep as is, add check before normalization) ---
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

    # Replace "-" with NaN before attempting conversion
    df_crux = df_crux.replace("-", np.nan)
    score_cols_crux = ["i@1", "i@5", "o@1", "o@5"]

    # Attempt to convert columns to numeric, coercing errors
    for col in score_cols_crux:
        if col in df_crux.columns:
            df_crux[col] = pd.to_numeric(df_crux[col], errors='coerce')
        else:
            logger.warning(f"Column '{col}' not found in Crux data.")
            # Optionally create the column with NaNs if needed later
            # df_crux[col] = np.nan


    new_df = pd.DataFrame()
    if "Model" not in df_crux.columns:
        logger.error("Crux data is missing the 'Model' column.")
        return pd.DataFrame(columns=["model_name", "score"])

    new_df["model_name"] = df_crux["Model"]

    valid_cols_for_norm_crux = [col for col in score_cols_crux if col in df_crux.columns and df_crux[col].notna().any()]

    if not valid_cols_for_norm_crux:
         logger.warning("No valid columns found for normalization in Crux.")
         new_df["score"] = np.nan
         return new_df[['model_name', 'score']]

    for col in valid_cols_for_norm_crux:
         normalize_column(df_crux, new_df, col) # Pass df_crux as source

    new_df["score"] = new_df[valid_cols_for_norm_crux].mean(axis=1)
    return new_df[['model_name', 'score']]


# --- fetch_tabby_leaderboard2 (keep as is, add check before normalization) ---
@lru_cache(maxsize=1)
def fetch_tabby_leaderboard2() -> pd.DataFrame:
    """
    Fetches Tabby leaderboard data from YAML, normalizes the scores, and calculates the mean score.
    """
    url = "https://leaderboard.tabbyml.com/tabby.yml"
    data = {} # Initialize data as empty dict
    try:
        response = SESSION.get(url, timeout=10) # Use the configured session
        response.raise_for_status()
        data = yaml.safe_load(response.text)
    except requests.exceptions.RequestException as e:
        logger.error("An error occurred while fetching Tabby data: %s", e)
    except yaml.YAMLError as e:
        logger.error("An error occurred while parsing Tabby YAML data: %s", e)
    except Exception as e: # Catch any other unexpected errors
        logger.error("An unexpected error occurred processing Tabby data: %s", e)

    if not data or not isinstance(data, dict): # Check if data is a non-empty dict
        logger.error("Fetched Tabby data is empty or not in expected dictionary format.")
        return pd.DataFrame(columns=["model_name", "score"])


    rows = []
    for model_name, model_data in data.items():
         # Check if model_data is a dictionary before iterating
         if not isinstance(model_data, dict):
             logger.warning(f"Skipping model '{model_name}' in Tabby data: invalid format '{type(model_data)}'.")
             continue
         for metric, scores in model_data.items():
             # Check if scores is a dictionary before iterating
             if not isinstance(scores, dict):
                 logger.warning(f"Skipping metric '{metric}' for model '{model_name}' in Tabby data: invalid scores format '{type(scores)}'.")
                 continue

             row = {"model_name": f"{model_name}_{metric.lower()}"}
             for lang, score in scores.items():
                 # Ensure lang is a string before lowercasing
                 if isinstance(lang, str):
                      row[f"{lang.lower()}_score"] = score
                 else:
                      logger.warning(f"Skipping language '{lang}' for metric '{metric}', model '{model_name}' in Tabby data: non-string key.")
             rows.append(row)


    if not rows:
        logger.warning("No valid rows generated for Tabby leaderboard.")
        return pd.DataFrame(columns=["model_name", "score"])

    df = pd.DataFrame(rows)

    # Identify score columns robustly
    score_columns = [col for col in df.columns if col.endswith("_score")]

    # Normalize only if score columns exist and have data
    valid_cols_for_norm_tabby = [col for col in score_columns if df[col].notna().any()]

    if not valid_cols_for_norm_tabby:
        logger.warning("No valid score columns found for normalization in Tabby.")
        df["score"] = np.nan
        return df[['model_name', 'score']]


    for col in valid_cols_for_norm_tabby:
         # Convert column to numeric before normalization
         df[col] = pd.to_numeric(df[col], errors='coerce')
         normalize_column(df, df, col) # Pass df as source

    df["score"] = df[valid_cols_for_norm_tabby].mean(axis=1)
    df = df.sort_values(by="score", ascending=False)
    df.reset_index(drop=True, inplace=True)
    return df[['model_name', 'score']]


# --- REVISED fetch_aider_leaderboard2 and get_leaderboard_df ---

def process_leaderboard(
        df: pd.DataFrame, model_col: str, edit_format_col: str, percent_col: str, format_col: str
) -> List[Dict[str, Any]]:
    """Processes the raw DataFrame from markdown into the desired score format."""
    rows = []
    required_cols = [model_col, edit_format_col, percent_col, format_col]
    if not all(col in df.columns for col in required_cols):
        logger.error(f"Missing required columns in input DataFrame for process_leaderboard. Required: {required_cols}, Found: {df.columns.tolist()}")
        return rows

    for _, row in df.iterrows():
        try:
            # Ensure values are strings before stripping, handle potential NaN/None
            model_val = str(row[model_col]) if pd.notna(row[model_col]) else ""
            edit_format_val = str(row[edit_format_col]) if pd.notna(row[edit_format_col]) else ""
            percent_val = str(row[percent_col]) if pd.notna(row[percent_col]) else ""
            format_val = str(row[format_col]) if pd.notna(row[format_col]) else ""

            model_name = f"{model_val.strip()}_{edit_format_val.strip()}"

            # Handle potential errors during percentage conversion
            try:
                percent_completed = float(percent_val.replace("%", "").strip())
            except ValueError:
                logger.warning(f"Could not convert '{percent_val}' to float for completed % in row: {row.to_dict()}")
                percent_completed = np.nan # Or assign 0, depending on desired handling

            try:
                percent_format = float(format_val.replace("%", "").strip())
            except ValueError:
                logger.warning(f"Could not convert '{format_val}' to float for format % in row: {row.to_dict()}")
                percent_format = np.nan # Or assign 0

            # Calculate score only if both percentages are valid numbers
            if pd.notna(percent_completed) and pd.notna(percent_format):
                score = (percent_completed + percent_format) / 200.0 # Ensure float division
                rows.append({"model_name": model_name, "score": score})
            else:
                 rows.append({"model_name": model_name, "score": np.nan}) # Append with NaN score if conversion failed

        except (AttributeError, KeyError) as e:
            logger.warning(f"Error processing row in process_leaderboard: {row.to_dict()}. Error: {e}")
            continue # Skip row on error
    return rows


def get_leaderboard_df(leaderboard_text: str, process_func: callable) -> pd.DataFrame:
    """
    Process leaderboard markdown text into DataFrame using more robust parsing.
    """
    if not leaderboard_text or not isinstance(leaderboard_text, str):
        logger.warning("Invalid or empty leaderboard text provided.")
        return pd.DataFrame(columns=["model_name", "score"])

    try:
        # Find the table header line
        header_match = re.search(r"^\s*\|\s*Model\s*\|\s*Edit format\s*\|.*Percent completed correctly.*\|", leaderboard_text, re.MULTILINE | re.IGNORECASE)
        if not header_match:
            logger.warning("Could not find header line in leaderboard data")
            return pd.DataFrame(columns=["model_name", "score"])

        header_line = header_match.group(0)
        table_start_index = header_match.end()

        # Find the separator line immediately after the header
        separator_match = re.search(r"^\s*\|?\s*-+\s*\|.*", leaderboard_text[table_start_index:], re.MULTILINE)
        if not separator_match:
            logger.warning("Could not find separator line after header")
            return pd.DataFrame(columns=["model_name", "score"])

        # Extract table content between header and the next non-table line/end of string
        table_content_match = re.search(r"^\s*\|?-.*?\n((?:^\s*\|.*?\n)*)", leaderboard_text[table_start_index:], re.MULTILINE | re.DOTALL)

        if not table_content_match:
             logger.warning("Could not extract table content rows.")
             return pd.DataFrame(columns=["model_name", "score"])

        data_lines_text = table_content_match.group(1)

        # Combine header and data lines for CSV parsing
        table_csv_text = header_line + "\n" + data_lines_text

        # Use StringIO to treat the string as a file
        buffer = io.StringIO(table_csv_text)

        # Read using pandas read_csv, skipping initial spaces and handling the pipe delimiter
        # The lineterminator is needed if rows might be split unexpectedly; adjust if necessary.
        # Using `on_bad_lines='warn'` or `'skip'` might be helpful if data is messy.
        df = pd.read_csv(
            buffer,
            sep="|",
            skipinitialspace=True, # Handles spaces after '|'
            index_col=False, # Don't use first column as index
            on_bad_lines='warn' # Report issues with rows having wrong number of fields
        )

        # Clean up DataFrame: remove empty leading/trailing columns and strip whitespace from headers/cells
        df = df.iloc[:, 1:-1] # Drop the first and last columns which are usually empty due to leading/trailing '|'
        df.columns = df.columns.str.strip()
        df = df.map(lambda x: x.strip() if isinstance(x, str) else x) # Strip whitespace from all string cells

        # Define expected columns for processing (case-insensitive match from header)
        # Find the actual column names corresponding to the logical roles
        model_col_name = next((col for col in df.columns if 'model' in col.lower()), None)
        edit_format_col_name = next((col for col in df.columns if 'edit format' in col.lower() and 'correct' not in col.lower()), None)
        percent_completed_col_name = next((col for col in df.columns if 'percent completed' in col.lower()), None)
        percent_format_col_name = next((col for col in df.columns if 'percent using correct' in col.lower()), None)

        required_cols_map = {
            "Model": model_col_name,
            "Edit format": edit_format_col_name,
            "Percent completed correctly": percent_completed_col_name,
            "Percent using correct edit format": percent_format_col_name
        }

        if not all(required_cols_map.values()):
            missing = [k for k, v in required_cols_map.items() if v is None]
            logger.error(f"Could not find all required columns in parsed Aider table. Missing: {missing}. Found columns: {df.columns.tolist()}")
            return pd.DataFrame(columns=["model_name", "score"])


        # Process the dataframe using the provided function
        result_rows = process_func(
            df,
            required_cols_map["Model"],
            required_cols_map["Edit format"],
            required_cols_map["Percent completed correctly"],
            required_cols_map["Percent using correct edit format"]
        )

        if not result_rows:
             logger.warning("Processing function returned no rows.")
             return pd.DataFrame(columns=["model_name", "score"])

        return pd.DataFrame(result_rows)

    except Exception as e:
        logger.error("Error processing leaderboard markdown: %s", e, exc_info=True) # Log traceback
        # Return empty DataFrame on any processing error
        return pd.DataFrame(columns=["model_name", "score"])


@lru_cache(maxsize=1)
def fetch_aider_leaderboard2(json_data_str: str = None) -> pd.DataFrame:
    """
    Fetches Aider leaderboard data, processes both old and new formats, and merges them.
    Accepts an optional JSON string for offline testing.
    """
    data = None
    if json_data_str:
        try:
            data = json.loads(json_data_str)
            logger.info("Using provided JSON string for Aider leaderboard.")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse provided JSON string: {e}")
            return pd.DataFrame(columns=["model_name", "score"])
    else:
        try:
            url = "https://aider.chat/assets/js/search-data.json"
            logger.info(f"Fetching Aider leaderboard data from {url}")
            response = SESSION.get(url, timeout=20) # Use session, increase timeout
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching Aider leaderboard from {url}: {e}")
            return pd.DataFrame(columns=["model_name", "score"])
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {url}: {e}")
            return pd.DataFrame(columns=["model_name", "score"])
        except Exception as e:
            logger.error(f"An unexpected error occurred fetching Aider data: {e}")
            return pd.DataFrame(columns=["model_name", "score"])

    if not data or not isinstance(data, dict):
        logger.error("Fetched Aider data is empty or not in the expected format.")
        return pd.DataFrame(columns=["model_name", "score"])

    # Find the content based on title - more robustly
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
             break # Stop searching once both are found

    if not old_leaderboard_content:
        logger.warning("Could not find 'Code editing leaderboard' content.")
    if not new_leaderboard_content:
         logger.warning("Could not find 'Aider polyglot coding leaderboard' content.")


    # Process leaderboards safely
    old_processed = pd.DataFrame(columns=["model_name", "score"])
    if old_leaderboard_content:
         logger.info("Processing old Aider leaderboard...")
         old_processed = get_leaderboard_df(old_leaderboard_content, process_leaderboard)
         if old_processed.empty:
              logger.warning("Old Aider leaderboard processing resulted in an empty DataFrame.")

    new_processed = pd.DataFrame(columns=["model_name", "score"])
    if new_leaderboard_content:
         logger.info("Processing new Aider leaderboard...")
         new_processed = get_leaderboard_df(new_leaderboard_content, process_leaderboard)
         if new_processed.empty:
             logger.warning("New Aider leaderboard processing resulted in an empty DataFrame.")

    # Handle cases where one or both dataframes might be empty
    if old_processed.empty and new_processed.empty:
        logger.warning("Both Aider leaderboards resulted in empty DataFrames.")
        return pd.DataFrame(columns=["model_name", "score"])
    elif old_processed.empty:
        logger.info("Using only new Aider leaderboard data.")
        return new_processed # Normalize might be needed if not done in get_leaderboard_df
    elif new_processed.empty:
        logger.info("Using only old Aider leaderboard data.")
        return old_processed # Normalize might be needed

    # Proceed with merge only if both have data
    logger.info("Merging old and new Aider leaderboards.")
    try:
        # Ensure 'score' columns are numeric before merge/mean calculation
        old_processed['score'] = pd.to_numeric(old_processed['score'], errors='coerce')
        new_processed['score'] = pd.to_numeric(new_processed['score'], errors='coerce')

        merged_df = pd.merge(old_processed, new_processed, on="model_name", how="outer", suffixes=("_old", "_new")) # Use suffixes

        # Calculate combined score using numpy for nan handling
        merged_df["score"] = np.nanmean(merged_df[["score_old", "score_new"]], axis=1)

        # Drop intermediate columns
        merged_df = merged_df.drop(["score_old", "score_new"], axis=1, errors="ignore")

        # Handle potential NaN scores after merge/mean
        merged_df = merged_df.dropna(subset=['score'])

        # Final check if empty after dropping NaNs
        if merged_df.empty:
            logger.warning("Merged Aider DataFrame is empty after dropping NaN scores.")

        return merged_df

    except Exception as e:
        logger.error(f"Error merging Aider leaderboards: {e}", exc_info=True)
        # Fallback to returning whichever dataframe is not empty, or an empty one
        if not new_processed.empty: return new_processed
        if not old_processed.empty: return old_processed
        return pd.DataFrame(columns=["model_name", "score"])


# --- (Keep existing merge_dataframes, get_leaderboard_score, get_merged_leaderboard_df) ---
def merge_dataframes(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Merges multiple dataframes of leaderboard scores into one, averaging scores for duplicate model names.
    Handles potential non-numeric scores gracefully.
    """
    if not dfs:
        return pd.DataFrame(columns=["model_name", "score"])

    # Concatenate all valid dataframes first
    all_dfs = [df for df in dfs if isinstance(df, pd.DataFrame) and not df.empty and "model_name" in df.columns and "score" in df.columns]
    if not all_dfs:
        return pd.DataFrame(columns=["model_name", "score"])

    merged_df = pd.concat(all_dfs, ignore_index=True)

    # Ensure score is numeric, coerce errors to NaN
    merged_df['score'] = pd.to_numeric(merged_df['score'], errors='coerce')

    # Group by model_name and calculate the mean score, dropping NaNs during mean calculation
    # Use .mean(skipna=True) implicitly by pandas groupby mean aggregation
    final_df = merged_df.groupby('model_name')['score'].mean().reset_index()

    # Drop rows where the final averaged score is NaN (optional, keeps only models with valid scores)
    final_df = final_df.dropna(subset=['score'])


    return final_df


@lru_cache(maxsize=1)
def get_merged_leaderboard_df(aider_json_str: str = None) -> pd.DataFrame:
    """
    Merges all leaderboard data sources into a single sorted dataframe.
    Accepts optional Aider JSON string for offline testing.
    """
    sources = []
    fetch_functions = {
        "BigCodeBench Easy": fetch_bigcodebench_leaderboard_easy,
        "BigCodeBench Hard": fetch_bigcodebench_leaderboard_hard,
        "EvalPlus": fetch_evalplus_leaderboard2,
        "Crux": fetch_crux_leaderboard2,
        "Tabby": fetch_tabby_leaderboard2,
        # Pass the aider_json_str only to the Aider fetch function
        "Aider": lambda: fetch_aider_leaderboard2(aider_json_str)
    }

    for name, func in fetch_functions.items():
        try:
            logger.info(f"Fetching {name} data...")
            df = func()
            if isinstance(df, pd.DataFrame) and not df.empty:
                 # Ensure required columns exist
                 if "model_name" in df.columns and "score" in df.columns:
                     # Keep only necessary columns and ensure score is numeric
                     df_cleaned = df[["model_name", "score"]].copy()
                     df_cleaned["score"] = pd.to_numeric(df_cleaned["score"], errors='coerce')
                     # Drop rows where score could not be converted
                     df_cleaned = df_cleaned.dropna(subset=['score'])
                     if not df_cleaned.empty:
                          sources.append(df_cleaned)
                          logger.info(f"Successfully processed {name} data ({len(df_cleaned)} rows).")
                     else:
                           logger.warning(f"{name} data resulted in empty DataFrame after cleaning.")
                 else:
                      logger.warning(f"{name} data is missing required columns ('model_name', 'score'). Found: {df.columns.tolist()}")

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

    # Sort and reset index only if the DataFrame is not empty
    merged_df.sort_values(by="score", ascending=False, inplace=True)
    merged_df.reset_index(drop=True, inplace=True)
    logger.info(f"Final merged DataFrame has {len(merged_df)} rows.")
    return merged_df


def get_leaderboard_score(model_name: str, aider_json_str: str = None) -> float:
    """
    Aggregates scores across all tests and leaderboards using normalized averaging.
    Accepts optional Aider JSON string for offline testing.
    """
    # Pass the JSON string down to the merging function
    merged_df = get_merged_leaderboard_df(aider_json_str=aider_json_str)

    # Ensure merged_df is valid before proceeding
    if merged_df.empty or 'model_name' not in merged_df.columns or 'score' not in merged_df.columns:
         logger.warning(f"Merged leaderboard is empty or invalid when searching for {model_name}.")
         return 0.0

    # Make matching case-insensitive for broader compatibility
    scores = merged_df[merged_df["model_name"].str.contains(model_name, case=False, na=False)] # Handle potential NaN in model_name

    if scores.empty:
        logger.info(f"No scores found containing '{model_name}' (case-insensitive).")
        return 0.0

    # Ensure score is numeric before mean calculation
    valid_scores = pd.to_numeric(scores["score"], errors='coerce').dropna()

    if valid_scores.empty:
        logger.warning(f"Found entries for '{model_name}', but scores were non-numeric or NaN.")
        return 0.0

    mean_score = float(valid_scores.mean())
    logger.info(f"Calculated mean score for '{model_name}': {mean_score:.4f} from {len(valid_scores)} entries.")
    return mean_score


if __name__ == "__main__":
    # --- Provide the JSON data as a string ---
    # WARNING: This is a very long string. In a real scenario, you might load it from a file.
    # For brevity in this example, I'll use a placeholder. Replace with your actual JSON string.
    aider_json_data_string = """
    {"0":{"doc":"Modify an open source 2048 game with aider","title":"Modify an open source 2048 game with aider","content":"..." },"1":{...}, ... ,"117":{"doc":"Code editing leaderboard","title":"Code editing leaderboard","content":"This old aider code editing leaderboard has been replaced by the new, much more challenging polyglot leaderboard.\\n\\nAider’s code editing benchmark asks the LLM to edit python source files to complete 133 small coding exercises from Exercism. This measures the LLM’s coding ability, and whether it can write new code that integrates into existing code. The model also has to successfully apply all its changes to the source file without human intervention.\\n\\n| Model | Percent completed correctly | Percent using correct edit format | Command | Edit format |\\n|---|---|---|---|---|\\n| o1 | 84.2% | 99.2% | aider --model openrouter/openai/o1 | diff |\\n| claude-3-5-sonnet-20241022 | 84.2% | 99.2% | aider --model anthropic/claude-3-5-sonnet-20241022 | diff |\\n| gemini-exp-1206 (whole) | 80.5% | 100.0% | aider --model gemini/gemini-exp-1206 | whole |\\n| o1-preview | 79.7% | 93.2% | aider --model o1-preview | diff |\\n| claude-3.5-sonnet-20240620 | 77.4% | 99.2% | aider --model claude-3.5-sonnet-20240620 | diff |\\n| claude-3-5-haiku-20241022 | 75.2% | 95.5% | aider --model anthropic/claude-3-5-haiku-20241022 | diff |\\n| ollama/qwen2.5-coder:32b | 72.9% | 100.0% | aider --model ollama/qwen2.5-coder:32b | whole |\\n| DeepSeek Coder V2 0724 | 72.9% | 97.7% | aider --model deepseek/deepseek-coder | diff |\\n| gpt-4o-2024-05-13 | 72.9% | 96.2% | aider | diff |\\n| DeepSeek-V2.5-1210 | 72.2% | 99.2% | aider --model deepseek/deepseek-chat | diff |\\n| openai/chatgpt-4o-latest | 72.2% | 97.0% | aider --model openai/chatgpt-4o-latest | diff |\\n| DeepSeek V2.5 | 72.2% | 96.2% | aider --deepseek | diff |\\n| gpt-4o-2024-11-20 | 71.4% | 99.2% | aider --model openai/gpt-4o-2024-11-20 | diff |\\n| Qwen2.5-Coder-32B-Instruct | 71.4% | 94.7% | aider --model openai/hf:Qwen/Qwen2.5-Coder-32B-Instruct --openai-api-base https://glhf.chat/api/openai/v1 | diff |\\n| gpt-4o-2024-08-06 | 71.4% | 98.5% | aider --model openai/gpt-4o-2024-08-06 | diff |\\n| o1-mini (whole) | 70.7% | 90.0% | aider --model o1-mini | whole |\\n| gemini-2.0-flash-exp | 69.9% | 97.0% | aider --model gemini/gemini-2.0-flash-exp | diff |\\n| DeepSeek Chat V2 0628 | 69.9% | 97.7% | aider --model deepseek/deepseek-chat | diff |\\n| gemini-exp-1206 (diff) | 69.2% | 84.2% | aider --model gemini/gemini-exp-1206 | diff |\\n| Qwen2.5-Coder-14B-Instruct | 69.2% | 100.0% | aider --model openai/Qwen2.5-Coder-14B-Instruct | whole |\\n| claude-3-opus-20240229 | 68.4% | 100.0% | aider --opus | diff |\\n| gpt-4-0613 | 67.7% | 100.0% | aider -4 | diff |\\n| Dracarys2-72B-Instruct | 66.9% | 100.0% | (via glhf.chat) | whole |\\n| gemini-1.5-pro-exp-0827 | 66.9% | 94.7% | aider --model gemini/gemini-1.5-pro-exp-0827 | diff-fenced |\\n| llama-3.1-405b-instruct (whole) | 66.2% | 100.0% | aider --model openrouter/meta-llama/llama-3.1-405b-instruct | whole |\\n| gpt-4-0314 | 66.2% | 93.2% | aider --model gpt-4-0314 | diff |\\n| gpt-4-0125-preview | 66.2% | 97.7% | aider --model gpt-4-0125-preview | udiff |\\n| yi-lightning | 65.4% | 97.0% | aider --model openai/yi-lightning | whole |\\n| openrouter/qwen/qwen-2.5-coder-32b-instruct | 65.4% | 84.2% | aider --model openrouter/qwen/qwen-2.5-coder-32b-instruct | diff |\\n| Mistral Large (2411) | 65.4% | 96.2% | aider --model mistral/mistral-large-latest | diff |\\n| gemini-1.5-pro-002 | 65.4% | 96.2% | aider --model gemini/gemini-1.5-pro-002 | diff-fenced |\\n| qwen-2.5-72b-instruct (bf16) | 65.4% | 96.2% | aider --model openrouter/qwen/qwen-2.5-72b-instruct | diff |\\n| gpt-4-1106-preview | 65.4% | 92.5% | aider --model gpt-4-1106-preview | udiff |\\n| ollama/Qwen2.5.1-Coder-7B-Instruct-GGUF:Q8_0-32k | 63.9% | 100.0% | aider --model ollama/Qwen2.5.1-Coder-7B-Instruct-GGUF:Q8_0-32k | whole |\\n| nousresearch/hermes-3-llama-3.1-405b | 63.9% | 100.0% | aider --model openrouter/nousresearch/hermes-3-llama-3.1-405b | whole |\\n| llama-3.1-405b-instruct (diff) | 63.9% | 92.5% | aider --model openrouter/meta-llama/llama-3.1-405b-instruct | diff |\\n| gpt-4-turbo-2024-04-09 (udiff) | 63.9% | 97.0% | aider --gpt-4-turbo | udiff |\\n| ollama/qwen2.5-coder:14b | 61.7% | 98.5% | aider --model ollama/qwen2.5-coder:14b | whole |\\n| o1-mini | 61.1% | 100.0% | aider --model o1-mini | diff |\\n| gemini-exp-1114 | 60.9% | 85.7% | aider --model gemini/gemini-exp-1114 | diff |\\n| Mistral Large 2 (2407) | 60.2% | 100.0% | aider --model mistral/mistral-large-2407 | whole |\\n| llama-3.3-70b-instruct | 59.4% | 88.7% | aider --model openrouter/meta-llama/llama-3.3-70b-instruct | diff |\\n| ollama/qwen2.5:32b-instruct-q8_0 | 58.6% | 100.0% | aider --model ollama/qwen2.5:32b-instruct-q8_0 | whole |\\n| Grok-2 | 58.6% | 98.5% | aider --model openrouter/x-ai/grok-2 | whole |\\n| llama-3.1-70b-instruct | 58.6% | 100.0% | aider --model fireworks_ai/accounts/fireworks/models/llama-v3p1-70b-instruct | whole |\\n| gemini-exp-1121 | 57.9% | 83.5% | aider --model gemini/gemini-exp-1121 | diff |\\n| Qwen2.5-Coder-7B-Instruct | 57.9% | 100.0% | aider --model openai/Qwen2.5-Coder-7B-Instruct | whole |\\n| gpt-3.5-turbo-0301 | 57.9% | 100.0% | aider --model gpt-3.5-turbo-0301 | whole |\\n| gpt-4-turbo-2024-04-09 (diff) | 57.6% | 100.0% | aider --model gpt-4-turbo-2024-04-09 | diff |\\n| gemini-1.5-pro-001 | 57.1% | 87.2% | aider --model gemini/gemini-1.5-pro-latest | diff-fenced |\\n| gpt-3.5-turbo-1106 | 56.1% | 100.0% | aider --model gpt-3.5-turbo-1106 | whole |\\n| gpt-4o-mini | 55.6% | 100.0% | aider --model gpt-4o-mini | whole |\\n| Qwen2 72B Instruct | 55.6% | 100.0% | aider --model together_ai/qwen/Qwen2-72B-Instruct | whole |\\n| Llama-3.1-Nemotron-70B-Instruct-HF | 54.9% | 99.2% | (via glhf.chat) | whole |\\n| Grok-2-mini | 54.9% | 100.0% | aider --model openrouter/x-ai/grok-2-mini | whole |\\n| claude-3-sonnet-20240229 | 54.9% | 100.0% | aider --sonnet | whole |\\n| Nova Pro | 54.1% | 100.0% | aider --model bedrock/us.amazon.nova-pro-v1:0 | whole |\\n| ollama/qwen2.5:32b | 54.1% | 100.0% | aider --model ollama/qwen2.5:32b | whole |\\n| Yi Coder 9B Chat | 54.1% | 100.0% | aider --model openai/hf:01-ai/Yi-Coder-9B-Chat --openai-api-base https://glhf.chat/api/openai/v1 | whole |\\n| gemini-1.5-flash-exp-0827 | 52.6% | 100.0% | aider --model gemini/gemini-1.5-flash-exp-0827 | whole |\\n| qwen2.5-coder:7b-instruct-q8_0 | 51.9% | 100.0% | aider --model ollama/qwen2.5-coder:7b-instruct-q8_0 | whole |\\n| gemini-1.5-flash-002 (0924) | 51.1% | 100.0% | aider --model gemini/gemini-1.5-flash-002 | whole |\\n| codestral-2405 | 51.1% | 100.0% | aider --model mistral/codestral-2405 | whole |\\n| gpt-3.5-turbo-0613 | 50.4% | 100.0% | aider --model gpt-3.5-turbo-0613 | whole |\\n| gpt-3.5-turbo-0125 | 50.4% | 100.0% | aider -3 | whole |\\n| qwen2:72b-instruct-q8_0 | 49.6% | 100.0% | aider --model ollama/qwen2:72b-instruct-q8_0 | whole |\\n| llama3-70b-8192 | 49.2% | 73.5% | aider --model groq/llama3-70b-8192 | diff |\\n| Codestral-22B-v0.1-Q4_K_M | 48.1% | 100.0% | aider --model Codestral-22B-v0.1-Q4_K_M | whole |\\n| codestral:22b-v0.1-q8_0 | 48.1% | 100.0% | aider --model ollama/codestral:22b-v0.1-q8_0 | whole |\\n| claude-3-haiku-20240307 | 47.4% | 100.0% | aider --model claude-3-haiku-20240307 | whole |\\n| ollama/codestral | 45.9% | 98.5% | aider --model ollama/codestral | whole |\\n| yi-coder:9b-chat-q4_0 | 45.1% | 100.0% | aider --model ollama/yi-coder:9b-chat-q4_0 | whole |\\n| gemini-1.5-flash-latest | 44.4% | 100.0% | aider --model gemini/gemini-1.5-flash-latest | whole |\\n| WizardLM-2 8x22B | 44.4% | 100.0% | aider --model openrouter/microsoft/wizardlm-2-8x22b | whole |\\n| ollama/yi-coder:9b-chat-fp16 | 43.6% | 99.2% | aider --model ollama/yi-coder:9b-chat-fp16 | whole |\\n| Reflection-70B | 42.1% | 100.0% | (not currently supported) | whole |\\n| Qwen2.5-Coder-3B-Instruct | 39.1% | 100.0% | aider --model openai/Qwen2.5-Coder-3B-Instruct | whole |\\n| ollama/mistral-small | 38.3% | 99.2% | aider --model ollama/mistral-small | whole |\\n| gemini-1.5-flash-8b-exp-0924 | 38.3% | 100.0% | aider --model gemini/gemini-1.5-flash-8b-exp-0924 | whole |\\n| Command R (08-24) | 38.3% | 100.0% | aider --model command-r-08-2024 | whole |\\n| Command R+ (08-24) | 38.3% | 100.0% | aider --model command-r-plus-08-2024 | whole |\\n| gemini-1.5-flash-8b-exp-0827 | 38.3% | 100.0% | aider --model gemini/gemini-1.5-flash-8b-exp-0827 | whole |\\n| llama-3.1-8b-instruct | 37.6% | 100.0% | aider --model fireworks_ai/accounts/fireworks/models/llama-v3p1-8b-instruct | whole |\\n| qwen1.5-110b-chat | 37.6% | 100.0% | aider --model together_ai/qwen/qwen1.5-110b-chat | whole |\\n| gemma2:27b-instruct-q8_0 | 36.1% | 100.0% | aider --model ollama/gemma2:27b-instruct-q8_0 | whole |\\n| codeqwen:7b-chat-v1.5-q8_0 | 34.6% | 100.0% | aider --model ollama/codeqwen:7b-chat-v1.5-q8_0 | whole |\\n| ollama/mistral-nemo:12b-instruct-2407-q4_K_M | 33.1% | 100.0% | aider --model ollama/mistral-nemo:12b-instruct-2407-q4_K_M | whole |\\n| ollama/codegeex4 | 32.3% | 97.0% | aider --model ollama/codegeex4 | whole |\\n| Qwen2.5-Coder-1.5B-Instruct | 31.6% | 100.0% | aider --model openai/Qwen2.5-Coder-1.5B-Instruct | whole |\\n| command-r-plus | 31.6% | 100.0% | aider --model command-r-plus | whole |\\n| ollama/hermes3:8b-llama3.1-fp16 | 30.1% | 98.5% | aider --model ollama/hermes3:8b-llama3.1-fp16 | whole |\\n| ollama/wojtek/opencodeinterpreter:6.7b | 30.1% | 91.0% | aider --model ollama/wojtek/opencodeinterpreter:6.7b | whole |\\n| o1-mini-2024-09-12 | 27.1% | 95.6% | aider --model o1-mini | whole |\\n| ollama/tulu3 | 26.3% | 100.0% | aider --model ollama/tulu3 | whole |\\n| ollama/llama3.2:3b-instruct-fp16 | 26.3% | 97.0% | aider --model ollama/llama3.2:3b-instruct-fp16 | whole |\\n| ollama/hermes3 | 22.6% | 98.5% | aider --model ollama/hermes3 | whole |\\n| ollama/granite3-dense:8b | 20.3% | 78.9% | aider --model ollama/granite3-dense:8b | whole |\\n| Qwen2.5-Coder-0.5B-Instruct | 14.3% | 100.0% | aider --model openai/Qwen2.5-Coder-0.5B-Instruct | whole |\\n\\n","url":"/docs/leaderboards/edit.html","relUrl":"/docs/leaderboards/edit.html"}, ... ,"192":{"doc":"Aider LLM Leaderboards","title":"Aider polyglot coding leaderboard","content":"View Select Detail ×\\n| | Model | Percent correct | Cost | Command | Correct edit format | Edit Format |\\n|---|---|---|---|---|---|---|\\n| ▶ | o3 (high) + gpt-4.1 | 82.7% | $69.29 | aider --model o3 --architect | 100.0% | architect |\\n| . | Dirname : 2025-04-17-01-20-35--o3-mini-high-diff-arch<br />  Test cases : 225<br />  Model : o3 (high) + gpt-4.1<br />  Edit format : architect<br />  Commit hash : 80909e1-dirty<br />  Editor model : gpt-4.1<br />  Editor edit format : editor-diff<br />  Pass rate 1 : 36.0<br />  Pass rate 2 : 82.7<br />  Pass num 1 : 81<br />  Pass num 2 : 186<br />  Percent cases well formed : 100.0<br />  Error outputs : 9<br />  Num malformed responses : 0<br />  Num with malformed responses : 0<br />  User asks : 166<br />  Lazy comments : 0<br />  Syntax errors : 0<br />  Indentation errors : 0<br />  Exhausted context windows : 0<br />  Test timeouts : 0<br />  Total tests : 225<br />  Command : aider --model o3 --architect<br />  Date : 2025-04-17<br />  Versions : 0.82.2.dev<br />  Seconds per case : 110.0<br />  Total cost : 69.2921 |\\n| ▶ | o3 (high) | 79.6% | $111.03 | aider --model o3 | 95.1% | diff |\\n| . | Dirname : 2025-04-16-21-20-55--o3-high-diff-temp0-exsys<br />  Test cases : 225<br />  Model : o3 (high)<br />  Edit format : diff<br />  Commit hash : 24805ff-dirty<br />  Pass rate 1 : 36.9<br />  Pass rate 2 : 79.6<br />  Pass num 1 : 83<br />  Pass num 2 : 179<br />  Percent cases well formed : 95.1<br />  Error outputs : 11<br />  Num malformed responses : 11<br />  Num with malformed responses : 11<br />  User asks : 110<br />  Lazy comments : 0<br />  Syntax errors : 0<br />  Indentation errors : 0<br />  Exhausted context windows : 0<br />  Test timeouts : 2<br />  Total tests : 225<br />  Command : aider --model o3<br />  Date : 2025-04-16<br />  Versions : 0.82.1.dev<br />  Seconds per case : 113.8<br />  Total cost : 111.0325 |\\n...(rest of the polyglot table)... |\\n\\nBy Paul Gauthier, last updated April 20, 2025.\\n","url":"/docs/leaderboards/#leaderboard-title","relUrl":"/docs/leaderboards/#leaderboard-title"} , ... }
    """
    # If you have the full JSON, paste it here. Otherwise, this placeholder will cause errors later.
    # For testing, ensure aider_json_data_string contains the actual JSON content provided in the prompt.
    # Due to length limits, I cannot paste the full JSON here again.

    import json # Make sure json is imported

    models = [
        "Magicoder-S-DS-6.7B",
        "StarCoder2-15B-Instruct-v0.1",
        "OpenCoder-8B-Instruct",
        "phi-1",
        "DeepSeekCoder-V2-Lite",
        "claude-3-5-sonnet-20241022", # Example from aider leaderboard
        "phi-1.5",
        "KwaiCoder-23B-A4B-v1",
        "Qwen-QWQ", # This is likely a specific variant, check exact names
        "gpt-4", # Broad term, might match multiple specific versions
        "gpt-4o", # Broad term
        "gpt-3.5", # Broad term
        "o1", # From aider leaderboard
        "gemini-exp-1206", # From aider leaderboard
        "DeepSeek Coder V2 0724", # From aider leaderboard
    ]

    model_scores = []
    for model in models:
        logger.info(f"\n--- Getting score for: {model} ---")
        # Pass the JSON string to the scoring function
        # Set to None if you want to test the live fetch: score = get_leaderboard_score(model, aider_json_str=None)
        score = get_leaderboard_score(model, aider_json_str=aider_json_data_string)
        if score > 0.0: # Check if score is positive (or handle 0.0 if it's valid)
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
    final_df = get_merged_leaderboard_df(aider_json_str=aider_json_data_string)
    if not final_df.empty:
         print(final_df.head(20).to_string())
    else:
         print("Merged DataFrame is empty.")