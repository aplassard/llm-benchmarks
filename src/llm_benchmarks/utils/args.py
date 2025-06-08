import argparse
from .prompt_utils import load_prompt, get_available_prompts

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
        choices=get_available_prompts(), # Dynamically set choices
        help="Keyword for the prompt to use. Corresponding <keyword>.txt file must exist in the prompts directory.",
        dest="prompt_keyword", # Store the keyword here
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

    # Load the prompt content using the keyword from args.prompt_keyword
    # and store it in args.prompt_template
    try:
        # args.prompt_keyword will exist due to dest="prompt_keyword"
        # The actual prompt content will be stored in args.prompt_template
        args.prompt_template = load_prompt(args.prompt_keyword)
    except FileNotFoundError:
        # This path should ideally not be hit if 'choices' and 'default' work as expected,
        # but it's a good safeguard.
        parser.error(
            f"Selected prompt file for keyword '{args.prompt_keyword}' not found. "
            f"Available prompts: {', '.join(get_available_prompts())}"
        )
        # The line below is effectively dead code if parser.error exits, which it does.
        # Keep for clarity if parser.error behavior changes or is mocked in a test.
        # raise  # Re-raise the FileNotFoundError or handle appropriately

    return args
