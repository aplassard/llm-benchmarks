import logging
import os
import uuid
from dotenv import load_dotenv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from .data import GSM8KDataset
from .solvers import GSM8KSolver, SolveResult
from .utils.args import parse_arguments
from .utils.logging import setup_logging
from llm_benchmarks.cache.cache import CacheManager

load_dotenv()


def process_example(example_data: dict, solver: GSM8KSolver, logger: logging.Logger, example_index: int) -> tuple[bool, bool, SolveResult | None]:
    """
    Processes a single example using the solver.
    Returns a tuple: (is_correct, was_skipped, solve_result_or_none).
    """
    question = example_data["question"]
    true_answer_full = example_data["answer"]

    result = solver.solve(question, true_answer_full)

    if result.extracted_true_answer is None:
        logger.warning(
            f"Could not extract ground truth answer for example {example_index + 1} (Q: {result.question[:50]}...). Skipping."
        )
        logger.debug(f"  Skipped Question: {result.question}")
        logger.debug(f"  Full Ground Truth for Skipped: {result.true_answer_full}")
        return False, True, result

    is_correct = False
    if result.extracted_model_answer is not None and \
       result.extracted_model_answer == result.extracted_true_answer:
        is_correct = True

    return is_correct, False, result


def run_benchmarks(args, logger, dataset, solver: GSM8KSolver):
    correct_answers = 0
    total_processed_valid_gt = 0 # Total examples where ground truth was extractable
    skipped_examples_gt_extraction_failed = 0

    num_to_run = (
        len(dataset)
        if args.num_examples == -1
        else min(args.num_examples, len(dataset))
    )

    if num_to_run == 0:
        logger.info("No examples to run. The dataset might be empty or num_examples is 0.")
        if total_processed_valid_gt == 0:
             logger.info("\nNo examples were processed that had extractable ground truth answers.")
        return

    logger.info(f"Running benchmark on {num_to_run} examples using {args.num_threads} threads...")

    futures_list = []
    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        for i in range(num_to_run):
            example = dataset[i]
            future = executor.submit(process_example, example, solver, logger, i)
            futures_list.append(future)

        for future in tqdm(as_completed(futures_list), total=len(futures_list), desc="Benchmarking Progress"):
            try:
                is_correct, was_skipped, solve_result = future.result() # solve_result can be used for more detailed logging if needed

                if was_skipped:
                    skipped_examples_gt_extraction_failed += 1
                    continue # Skip to next future, already logged in process_example

                total_processed_valid_gt +=1 # Count only if GT was extractable

                if is_correct:
                    correct_answers += 1
            except Exception as e:
                logger.error(f"An error occurred while processing an example: {e}", exc_info=True)


    if total_processed_valid_gt > 0:
        accuracy = (correct_answers / total_processed_valid_gt) * 100
        logger.info("\n--- Benchmark Summary ---")
        logger.info(f"Total examples attempted: {num_to_run}")
        logger.info(f"Examples skipped (GT extraction failed): {skipped_examples_gt_extraction_failed}")
        logger.info(f"Total examples with valid ground truth: {total_processed_valid_gt}")
        logger.info(f"Correct answers: {correct_answers}")
        logger.info(f"Accuracy (based on valid GT examples): {accuracy:.2f}%")
    else:
        logger.info("\nNo examples were processed that had extractable ground truth answers.")
    logger.info(f"Run ID: {solver.run_id if hasattr(solver, 'run_id') else 'N/A (caching disabled or not set)'}")


def main():
    args = parse_arguments()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    if not os.getenv("OPENROUTER_API_KEY"):
        logger.error("OPENROUTER_API_KEY environment variable not set.")
        logger.error("Please set it in your .env file or environment.")
        return

    run_id = str(uuid.uuid4())
    logger.info(f"Current Run ID: {run_id}")
    
    prompt_template_content = args.prompt_template
    prompt_template_name = args.prompt
    logger.info(f"Using prompt template file: {prompt_template_name} (Name for cache: {prompt_template_name})")

    if "{content}" not in prompt_template_content:
        logger.error(f"Prompt template {args.prompt_template} must contain a '{{content}}' placeholder.")
        return

    logger.info(f"Using model: {args.model_name}")
    logger.info(f"Loading GSM8K dataset (split: {args.data_split}, config: {args.data_config})...")
    try:
        dataset = GSM8KDataset(split=args.data_split, config=args.data_config)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    # Ensure solver is initialized before run_benchmarks is called.
    # The existing logic for solver initialization based on args.no_cache is maintained.

    solver: GSM8KSolver

    if args.no_cache:
        logger.info("Caching is disabled by --no-cache flag.")
        logger.info(f"Initializing GSM8KSolver with model: {args.model_name} (caching disabled)...")
        solver = GSM8KSolver(
            model_name=args.model_name,
            prompt_template=prompt_template_content,
            cache_manager=None,
            prompt_template_name=prompt_template_name, # Still useful for potential non-cache logging/identification
            data_split=args.data_split, # For context if needed, though cache is off
            data_config=args.data_config, # For context
            run_id=run_id # Pass run_id even if caching is off for consistency
        )
        run_benchmarks(args, logger, dataset, solver)
    else:
        cache_db_path = "llm_benchmarks_cache.sqlite3"
        logger.info(f"Cache database path: {cache_db_path}")
        logger.info(f"Cache usage enabled (default or --cache flag used).")

        with CacheManager(db_path=cache_db_path, use_cache=True) as cache_manager: # Ensure DB is initialized
            cache_manager.init_db() # Create tables if they don't exist

            logger.info(f"Initializing GSM8KSolver with model: {args.model_name} (caching enabled)...")
            solver = GSM8KSolver(
                model_name=args.model_name,
                prompt_template=prompt_template_content,
                cache_manager=cache_manager,
                prompt_template_name=prompt_template_name, # For cache key generation
                data_split=args.data_split, # For cache key generation
                data_config=args.data_config, # For cache key generation
                run_id=run_id # Pass current run_id for cache metadata
            )
            run_benchmarks(args, logger, dataset, solver)


if __name__ == "__main__":
    main()
