import argparse
from .prompt_utils import load_prompt, get_available_prompts

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="LLM Benchmarking CLI")
    parser.add_argument(
        "--model-name",
        type=str,
        default="mistralai/mistral-7b-instruct",
        help="Name of the model to use from OpenRouter.",
        dest="model_name",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="default",
        choices=get_available_prompts(),
        help="Keyword for the prompt to use. Corresponding <keyword>.txt file must exist in the prompts directory.",
        dest="prompt",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=10,
        help="Number of examples to run from the test set. Set to -1 to run all.",
        dest="num_examples",
    )
    parser.add_argument(
        "--data-split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Dataset split to use for benchmarking.",
        dest="data_split",
    )
    parser.add_argument(
        "--data-config",
        type=str,
        default="main",
        choices=["main", "socratic"],
        help="Dataset configuration to use.",
        dest="data_config",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level. Must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL.",
        dest="log_level",
    )
    # New argument
    parser.add_argument(
        "--no-cache",
        action="store_true",
        # Default is False when action='store_true' and flag is not present.
        # Explicitly setting default=False is not strictly needed but is clear.
        help="Disable caching of results.",
        dest="no_cache", # Ensure 'dest' is specified if you want to access it as args.no_cache
    )
    args = parser.parse_args()

    try:
        args.prompt_template = load_prompt(args.prompt)
    except FileNotFoundError:
        parser.error(
            f"Selected prompt file for keyword '{args.prompt}' not found. "
            f"Available prompts: {', '.join(get_available_prompts())}"
        )

    return args
