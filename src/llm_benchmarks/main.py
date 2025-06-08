import logging
import os
import uuid # For run_id
from dotenv import load_dotenv
from tqdm import tqdm

from .data import GSM8KDataset
from .solvers import GSM8KSolver
from .utils.args import parse_arguments
from .utils.logging import setup_logging
from llm_benchmarks.cache.cache import CacheManager # For caching

# Load environment variables from .env file
load_dotenv()


def main():
    args = parse_arguments()

    setup_logging(args.log_level)
    
    logger = logging.getLogger(__name__)

    if not os.getenv("OPENROUTER_API_KEY"):
        logger.error("OPENROUTER_API_KEY environment variable not set.")
        logger.error("Please set it in your .env file or environment.")
        return

    # --- Caching Setup ---
    run_id = str(uuid.uuid4())
    logger.info(f"Current Run ID: {run_id}")

    cache_db_path = "llm_benchmarks_cache.sqlite3"
    logger.info(f"Cache database path: {cache_db_path}")
    logger.info(f"Cache usage enabled: {not args.no_cache}")

    with CacheManager(db_path=cache_db_path, use_cache=not args.no_cache) as cache_manager:
        cache_manager.init_db()

        logger.info(f"Using model: {args.model_name}")
        logger.info(
            f"Loading GSM8K dataset (split: {args.data_split}, config: {args.data_config})..."
        )
        try:
            dataset = GSM8KDataset(split=args.data_split, config=args.data_config)
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return

        prompt_template_name = os.path.basename(args.prompt_template)
        logger.info(f"Using prompt template file: {args.prompt_template} (Name for cache: {prompt_template_name})")

        try:
            with open(args.prompt_template, "r", encoding="utf-8") as f:
                prompt_template_content = f.read()
        except FileNotFoundError:
            logger.error(f"Prompt template file not found: {args.prompt_template}")
            return
        except Exception as e:
            logger.error(f"Error reading prompt template file {args.prompt_template}: {e}")
            return

        if "{content}" not in prompt_template_content:
            logger.error(f"Prompt template {args.prompt_template} must contain a '{{content}}' placeholder.")
            return

        logger.info(f"Initializing GSM8KSolver with model: {args.model_name}...")
        solver = GSM8KSolver(
            model_name=args.model_name,
            prompt_template=prompt_template_content,  # Pass the loaded content
            cache_manager=cache_manager,
            prompt_template_name=prompt_template_name,
            data_split=args.data_split,
            data_config=args.data_config,
            run_id=run_id
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
            # Still print summary outside the loop, so return here if no examples
            if total_examples == 0:
                 logger.info("\nNo examples were processed that had extractable ground truth answers.")
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

            total_examples +=1

            if result.extracted_model_answer is not None and \
               result.extracted_model_answer == result.extracted_true_answer:
                correct_answers += 1

        # cache_manager.close() is called automatically by __exit__ when 'with' block ends

    # Summary printout
    if total_examples > 0:
        accuracy = (correct_answers / total_examples) * 100
        logger.info("\n--- Benchmark Summary ---")
        logger.info(f"Total examples processed: {total_examples}")
        logger.info(f"Correct answers: {correct_answers}")
        logger.info(f"Accuracy: {accuracy:.2f}%")
    else:
        # This case is now handled if num_to_run is 0 or if all examples were skipped.
        logger.info("\nNo examples were processed that had extractable ground truth answers.")


if __name__ == "__main__":
    main()
