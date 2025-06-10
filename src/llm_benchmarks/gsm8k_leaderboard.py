import sqlite3
import pandas as pd
import json
import logging
import streamlit as st

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_token_usage(json_str):
    """
    Extracts token usage from a JSON string.

    Args:
        json_str: A JSON string, typically from the 'model_full_response_json' column.

    Returns:
        A tuple (prompt_tokens, completion_tokens, total_tokens).
        Returns (0, 0, 0) if 'usage' or tokens are not found or if json_str is None.
    """
    if json_str is None:
        return 0, 0, 0
    try:
        data = json.loads(json_str)
        usage = data.get('usage', {})
        prompt_tokens = usage.get('prompt_tokens', 0)
        completion_tokens = usage.get('completion_tokens', 0)
        total_tokens = usage.get('total_tokens', 0)
        return prompt_tokens, completion_tokens, total_tokens
    except (json.JSONDecodeError, AttributeError):
        logging.warning(f"Could not parse JSON or find usage data in: {json_str}", exc_info=True)
        return 0, 0, 0

def load_and_process_data(db_path="llm_benchmarks_cache.sqlite3"):
    """
    Loads data from the SQLite database, processes it, and generates a leaderboard.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        A Pandas DataFrame with the aggregated leaderboard data.
        Returns an empty DataFrame if an error occurs.
    """
    try:
        conn = sqlite3.connect(db_path)
        # Check if table exists
        table_check = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table' AND name='results';", conn)
        if table_check.empty:
            logging.warning(f"Table 'results' not found in database {db_path}.")
            conn.close()
            return pd.DataFrame()

        df = pd.read_sql_query("SELECT * FROM results", conn)
        conn.close()
    except sqlite3.Error as e:
        logging.error(f"SQLite error: {e}")
        return pd.DataFrame()

    if df.empty:
        logging.info("No data found in the results table.")
        return pd.DataFrame()

    # Apply extract_token_usage
    tokens_df = df['model_full_response_json'].apply(lambda x: pd.Series(extract_token_usage(x), index=['prompt_tokens', 'completion_tokens', 'total_tokens']))
    df = pd.concat([df, tokens_df], axis=1)

    # Create is_correct column
    df['is_correct'] = (
        df['model_extracted_answer'].notna() &
        df['dataset_extracted_answer'].notna() &
        (df['model_extracted_answer'].astype(str) == df['dataset_extracted_answer'].astype(str)) # Cast to string for robust comparison
    )

    # Group and aggregate
    grouped_df = df.groupby(['model_name', 'prompt_template_name']).agg(
        num_entries=('eval_id', 'count'),  # <-- This is the line to change
        correct_count=('is_correct', 'sum'),
        total_prompt_tokens=('prompt_tokens', 'sum'),
        total_completion_tokens=('completion_tokens', 'sum'),
        total_tokens_all_entries=('total_tokens', 'sum')
    ).reset_index()

    # Calculate accuracy
    grouped_df['accuracy'] = grouped_df.apply(
        lambda row: row['correct_count'] / row['num_entries'] if row['num_entries'] > 0 else 0,
        axis=1
    )

    # Calculate average_tokens_per_entry
    grouped_df['average_tokens_per_entry'] = grouped_df.apply(
        lambda row: row['total_tokens_all_entries'] / row['num_entries'] if row['num_entries'] > 0 else 0,
        axis=1
    )

    # Select and rename columns
    leaderboard_df = grouped_df.rename(columns={
        'prompt_template_name': 'prompt_name',
        'total_tokens_all_entries': 'total_tokens_used',
        'total_prompt_tokens': 'total_prompt_tokens_used',
        'total_completion_tokens': 'total_completion_tokens_used'
    })

    final_columns = [
        'model_name',
        'prompt_name',
        'num_entries',
        'accuracy',
        'total_tokens_used',
        'total_prompt_tokens_used',
        'total_completion_tokens_used',
        'average_tokens_per_entry'
    ]
    # Ensure all expected columns exist, add if missing (e.g. if no data for token usage)
    for col in final_columns:
        if col not in leaderboard_df.columns:
            leaderboard_df[col] = 0 # Or pd.NA or appropriate default

    leaderboard_df = leaderboard_df[final_columns]

    return leaderboard_df

def main():
    st.set_page_config(page_title="GSM8K Leaderboard", layout="wide")
    st.title("GSM8K Model Performance Leaderboard")

    # Allow user to specify database path via text input for flexibility
    db_file_path = st.text_input(
        "Enter path to benchmark database:",
        "llm_benchmarks_cache.sqlite3"
    )

    if not db_file_path:
        st.info("Please enter a path to the database file.")
        st.stop()

    df_leaderboard = load_and_process_data(db_path=db_file_path)

    if df_leaderboard is None or df_leaderboard.empty:
        st.warning(f"No data available in the cache at '{db_file_path}'. Run the benchmark first or check the path.")
        st.stop()

    st.sidebar.header("Filters")

    # Model Name Filter
    all_model_names = sorted(df_leaderboard["model_name"].unique())
    selected_model_names = st.sidebar.multiselect(
        "Filter by Model Name",
        options=all_model_names,
        default=all_model_names
    )

    # Prompt Name Filter
    all_prompt_names = sorted(df_leaderboard["prompt_name"].unique())
    selected_prompt_names = st.sidebar.multiselect(
        "Filter by Prompt Name",
        options=all_prompt_names,
        default=all_prompt_names
    )

    # Number of Entries Filter
    min_entries = int(df_leaderboard["num_entries"].min())
    max_entries = int(df_leaderboard["num_entries"].max())

    if min_entries == max_entries:
        max_entries = min_entries + 1 # Ensure slider has a range

    selected_min_entries = st.sidebar.slider(
        "Filter by Minimum Number of Entries",
        min_value=min_entries,
        max_value=max_entries,
        value=min_entries
    )

    # Apply filters
    filtered_df = df_leaderboard[
        df_leaderboard["model_name"].isin(selected_model_names) &
        df_leaderboard["prompt_name"].isin(selected_prompt_names) &
        (df_leaderboard["num_entries"] >= selected_min_entries)
    ]

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **Column Explanations:**
    - **model_name**: Name of the language model.
    - **prompt_name**: Identifier for the prompt template used.
    - **num_entries**: Number of questions evaluated for this combination.
    - **accuracy**: Percentage of correctly answered questions.
    - **total_tokens_used**: Sum of all tokens (prompt + completion) for these entries.
    - **total_prompt_tokens_used**: Sum of prompt tokens.
    - **total_completion_tokens_used**: Sum of completion tokens.
    - **average_tokens_per_entry**: Average total tokens used per question.
    """)

    st.header("Leaderboard Results")
    if filtered_df.empty:
        st.info("No data matches the current filter criteria.")
    else:
        # Displaying with st.dataframe for better default formatting and interactivity
        st.dataframe(
            filtered_df.style.format({"accuracy": "{:.2%}", "average_tokens_per_entry": "{:.2f}"}),
            use_container_width=True
        )

if __name__ == "__main__":
    main()
