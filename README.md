# LLM Benchmarks for GSM8K

This project provides tools to benchmark Large Language Models (LLMs) on the GSM8K dataset and view the results.

## Features

-   Run benchmarks against various LLMs for the GSM8K dataset.
-   Cache results locally in an SQLite database (`llm_benchmarks_cache.sqlite3`).
-   View a Streamlit-based leaderboard to compare model performance.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd llm-benchmarks
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate # On Windows use .venv\Scripts\activate
    ```

3.  **Install dependencies:**
    The project uses `setuptools` for packaging. Install the package and its dependencies, including the command-line scripts:
    ```bash
    pip install -e .
    ```
    This will install `pandas`, `streamlit`, `openai`, and other necessary libraries.

## Running the Benchmark

To run the GSM8K benchmark and populate the results cache:

1.  **Ensure you have your OpenAI API key set up.** You can typically do this by setting the `OPENAI_API_KEY` environment variable. Create a `.env` file in the project root with your key if needed:
    ```
    OPENAI_API_KEY="your_api_key_here"
    ```
    The application uses `python-dotenv` to load this.

2.  **Execute the benchmark runner script:**
    ```bash
    gsm8k-benchmark-runner [options]
    ```
    This script will fetch questions from the GSM8K dataset, query the specified language models, extract answers, and store all information in the `llm_benchmarks_cache.sqlite3` database in the current directory.

    Refer to the script's help for available options:
    ```bash
    gsm8k-benchmark-runner --help
    ```

## Viewing the Leaderboard

After running the benchmarks, you can view the performance leaderboard.

To start the leaderboard, navigate to the project's root directory and execute the following command:
```bash
streamlit run src/llm_benchmarks/gsm8k_leaderboard.py
```
This will start the Streamlit application, and you can typically view it in your web browser at `http://localhost:8501`.

The leaderboard displays aggregated results from the `llm_benchmarks_cache.sqlite3` file, allowing you to filter and compare different models and prompt templates. Ensure this database file exists and contains data from a benchmark run.

## Development

(Optional: Add notes about linters, tests, etc. if relevant for contributors)

-   This project uses `ruff` for linting and `pytest` for testing.
-   Pre-commit hooks are configured to run checks automatically.

```bash
pre-commit install
pytest
```

This will ensure your contributions adhere to the project's coding standards.
