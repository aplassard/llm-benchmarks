import logging # Still needed for logger = logging.getLogger(__name__)
import os
# import re # No longer needed
from dotenv import load_dotenv
from tqdm import tqdm # Still needed for progress bar

from .data import GSM8KDataset
from .solvers import GSM8KSolver
from .utils.args import parse_arguments
from .utils.logging import setup_logging # Import the new logging setup function

# Load environment variables from .env file
load_dotenv()

# TqdmLoggingHandler class definition removed

def main():
    args = parse_arguments() # Call the new function for argument parsing

    # Call the new logging setup function
    setup_logging(args.log_level)
    
    # Get a logger for the main module, after setup_logging has configured the root logger
    logger = logging.getLogger(__name__)

    # Logging level info message is now part of setup_logging, or can be emitted here if preferred
    # logger.info(f"Logging level set to {args.log_level}") # This is implicitly handled by setup_logging

    if not os.getenv("OPENROUTER_API_KEY"):
        logger.error("OPENROUTER_API_KEY environment variable not set.")
        logger.error("Please set it in your .env file or environment.")
        return

    # Example of using logger after setup
    logger.info(f"Using model: {args.model_name}")
    logger.info(
        f"Loading GSM8K dataset (split: {args.data_split}, config: {args.data_config})..."
    )
    try:
        dataset = GSM8KDataset(split=args.data_split, config=args.data_config)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    logger.info(f"Initializing GSM8KSolver with model: {args.model_name}...")
    solver = GSM8KSolver(
        model_name=args.model_name,
        prompt_template=args.prompt_template,
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
            # We can use logger.debug for information previously shown with verbose flag
            logger.debug(f"  Skipped Question: {result.question}")
            logger.debug(f"  Full Ground Truth for Skipped: {result.true_answer_full}")
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
