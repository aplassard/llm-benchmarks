import argparse

DEFAULT_PROMPT_TEMPLATE = """Question: {content}

Please provide a step-by-step solution and end your response with the phrase 'The final answer is X' where X is the numerical answer."""

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
        "--prompt-template",  # Changed from prompt_template
        type=str,
        default=DEFAULT_PROMPT_TEMPLATE,
        help="Prompt template to use. Must include '{content}'.",
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
    return parser.parse_args()
