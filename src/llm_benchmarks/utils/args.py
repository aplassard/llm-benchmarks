import argparse
import os

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="LLM Benchmarking CLI")
    parser.add_argument(
        "--model-name",  # Changed from model_name
        type=str,
        default="mistralai/mistral-7b-instruct",
        help="Name of the model to use from OpenRouter.",
        dest="model_name",  # Keep dest as model_name
    )
    parser.add_argument(
        "--prompt-keyword",
        type=str,
        default="default",
        help="Keyword for the prompt file (e.g., 'default', 'rigorous'). The file should be in `src/llm_benchmarks/utils/prompts/` and named `<keyword>.txt`.",
        dest="prompt_template",  # Keep dest as prompt_template
    )
    parser.add_argument(
        "--num-examples",  # Changed from num_examples
        type=int,
        default=10,
        help="Number of examples to run from the test set. Set to -1 to run all.",
        dest="num_examples",  # Keep dest as num_examples
    )
    parser.add_argument(
        "--data-split",  # Changed from data_split
        type=str,
        default="test",
        choices=["train", "test"],
        help="Dataset split to use for benchmarking.",
        dest="data_split",  # Keep dest as data_split
    )
    parser.add_argument(
        "--data-config",  # Changed from data_config
        type=str,
        default="main",
        choices=["main", "socratic"],
        help="Dataset configuration to use.",
        dest="data_config",  # Keep dest as data_config
    )
    # --verbose argument removed
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], # Reinstate choices
        help="Set the logging level. Must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL.", # Update help
        dest="log_level",
    )
    args = parser.parse_args()

    # Load prompt content from file based on keyword
    keyword = args.prompt_template
    # Construct the path relative to the script location or a known base directory.
    # Assuming this script is run from a location where 'src/llm_benchmarks/utils/prompts/' is a valid relative path.
    # For robustness, consider using absolute paths or paths relative to this file's location.
    prompt_file_path = os.path.join("src", "llm_benchmarks", "utils", "prompts", f"{keyword}.txt")

    try:
        with open(prompt_file_path, "r", encoding="utf-8") as f:
            args.prompt_template = f.read()
    except FileNotFoundError:
        parser.error(f"Prompt file not found: {prompt_file_path}. Please ensure a file named '{keyword}.txt' exists in the 'src/llm_benchmarks/utils/prompts/' directory.")
        # Alternatively, raise an error or handle it as appropriate for your application
        # raise FileNotFoundError(f"Prompt file not found: {prompt_file_path}")

    return args
