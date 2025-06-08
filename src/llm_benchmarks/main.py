import argparse
import logging
import os
# import re # No longer needed
from dotenv import load_dotenv
from tqdm import tqdm

from .data import GSM8KDataset  # Adjusted import
from .solvers import GSM8KSolver  # Adjusted import

# Load environment variables from .env file
load_dotenv()

DEFAULT_PROMPT_TEMPLATE = """Question: {content}

Please provide a step-by-step solution and end your response with the phrase 'The final answer is X' where X is the numerical answer."""


class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


def main():
    # Configure logging
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Add the handler to the root logger
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    
    # Check if a handler is already present to avoid duplicates
    if not root_logger.handlers:
        root_logger.addHandler(handler)
    
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Run GSM8K benchmark.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="mistralai/mistral-7b-instruct",
        help="Name of the model to use from OpenRouter.",
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default=DEFAULT_PROMPT_TEMPLATE,
        help="Prompt template to use. Must include '{content}'.",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=10,
        help="Number of examples to run from the test set. Set to -1 to run all.",
    )
    parser.add_argument(
        "--data_split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Dataset split to use for benchmarking.",
    )
    parser.add_argument(
        "--data_config",
        type=str,
        default="main",
        choices=["main", "socratic"],
        help="Dataset configuration to use.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information for each example.",
    )

    args = parser.parse_args()

    if not os.getenv("OPENROUTER_API_KEY"):
        logger.error("OPENROUTER_API_KEY environment variable not set.")
        logger.error("Please set it in your .env file or environment.")
        return

    logger.info(
        f"Loading GSM8K dataset (split: {args.data_split}, config: {args.data_config})..."
    )
    try:
        dataset = GSM8KDataset(split=args.data_split, config=args.data_config)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    logger.info(f"Initializing GSM8KSolver with model: {args.model_name}...")
    solver = GSM8KSolver( # MODIFIED - pass verbose
        model_name=args.model_name,
        prompt_template=args.prompt_template,
        verbose=args.verbose, # NEW
    )

    correct_answers = 0
    total_examples = 0

    num_to_run = (
        len(dataset)
        if args.num_examples == -1
        else min(args.num_examples, len(dataset))
    )

    if num_to_run == 0:
        logger.info("No examples to run. The dataset might be empty or num_examples is 0.")
        return

    logger.info(f"Running benchmark on {num_to_run} examples...")

    for i in tqdm(range(num_to_run), desc="Benchmarking Progress"):
        example = dataset[i]
        question = example["question"]
        true_answer_full = example["answer"]

        # The true answer extraction will now be handled by the solver or post-processing of GSM8KResult
        # For now, we expect the solver to provide this if needed, or we adapt the loop later.
        # true_answer_extracted = extract_gsm8k_answer(true_answer_full) # This line is removed

        # Placeholder for where true_answer_extracted would be used or checked.
        # This logic will need to be updated based on how GSM8KResult is used.
        # For this subtask, we are only moving the function.
        # The direct usage of true_answer_extracted here will be addressed in a future step.
        # For now, we'll assume it might be None and adapt the condition slightly,
        # acknowledging this part of main() will change.

        # Simulating that we might not have it directly here anymore for the check.
        # This part of the code (main function) will be updated in a subsequent task
        # to correctly use the GSM8KResult object returned by solver.solve().
        # For now, to make the file syntactically valid after removing extract_gsm8k_answer,
        # we'll have to adjust or temporarily comment out its direct usage.
        # For the purpose of this step (moving the function), we will remove the direct call
        # and the immediate check. The actual handling of results will be in a later task.

        # if true_answer_extracted is None: # This check will need to be re-evaluated
        # print(
        # "Warning: Could not extract ground truth answer for example {i+1}. Skipping."
        # )
        # if args.verbose:
        # print(f"  Question: {question}")
        # print(f"  Full Ground Truth: {true_answer_full}")
        # continue
        # total_examples += 1 # This also needs to be handled carefully

        # Call the solver - MODIFIED
        result = solver.solve(question, true_answer_full)

        # Check if true answer could be extracted (using result object) - MODIFIED
        if result.extracted_true_answer is None:
            # total_examples was incremented before this check in the previous version,
            # which was incorrect. We should only count examples where ground truth is extractable.
            # So, we print the warning and skip.
            logger.warning(
                f"Could not extract ground truth answer for example {i+1} (Q: {result.question[:50]}...). Skipping."
            )
            # The solver handles its own verbose printing for the attempt.
            # This verbose block is for main.py's context of *skipping*.
            if args.verbose:
                logger.info(f"  Skipped Question: {result.question}")
                logger.info(f"  Full Ground Truth for Skipped: {result.true_answer_full}")
            continue

        # If we reach here, ground truth was extractable.
        total_examples += 1

        # Verbose printing for individual example processing is REMOVED from here.
        # It's now handled by the solver based on the verbose flag passed to it.

        # Compare answers (using result object) - MODIFIED
        # Correctness requires both extracted_model_answer and extracted_true_answer to be non-None and equal.
        # extracted_true_answer is confirmed non-None by the check above.
        if result.extracted_model_answer is not None and \
           result.extracted_model_answer == result.extracted_true_answer:
            correct_answers += 1

    if total_examples > 0:
        accuracy = (correct_answers / total_examples) * 100
        logger.info("\n--- Benchmark Summary ---")
        logger.info(f"Total examples processed: {total_examples}")
        logger.info(f"Correct answers: {correct_answers}")
        logger.info(f"Accuracy: {accuracy:.2f}%")
    else:
        logger.info("\nNo examples were processed that had extractable ground truth answers.")


if __name__ == "__main__":
    main()
