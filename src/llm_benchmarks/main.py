import logging
import os
from dotenv import load_dotenv
from tqdm import tqdm

from .data import GSM8KDataset
from .solvers import GSM8KSolver
from .utils.args import parse_arguments
from .utils.logging import setup_logging

# Load environment variables from .env file
load_dotenv()


def main():
    args = parse_arguments()

    setup_logging(args.log_level)
    
    # Get a logger for the main module, after setup_logging has configured the root logger
    logger = logging.getLogger(__name__)

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

        result = solver.solve(question, true_answer_full)

        if result.extracted_true_answer is None:
            logger.warning(
                f"Could not extract ground truth answer for example {i+1} (Q: {result.question[:50]}...). Skipping."
            )
            logger.debug(f"  Skipped Question: {result.question}")
            logger.debug(f"  Full Ground Truth for Skipped: {result.true_answer_full}")
            continue

        # If we reach here, ground truth was extractable.
        total_examples += 1

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
