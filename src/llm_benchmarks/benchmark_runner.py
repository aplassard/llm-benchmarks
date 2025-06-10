import logging
import os
import uuid # For run_id
import asyncio # Added asyncio
from dotenv import load_dotenv
from tqdm import tqdm

from .data import GSM8KDataset
from .solvers import GSM8KSolver, GSM8KResult # Added GSM8KResult
from .utils.args import parse_arguments
from .utils.logging import setup_logging
from llm_benchmarks.cache.cache import CacheManager # For caching

# Load environment variables from .env file
load_dotenv()


async def run_benchmarks(args, logger, dataset, solver: GSM8KSolver): # Changed to async def, added type hint
    correct_answers = 0
    total_examples = 0
    tasks = []

    num_to_run = (
        len(dataset)
        if args.num_examples == -1
        else min(args.num_examples, len(dataset))
    )

    if num_to_run == 0:
        logger.info("No examples to run. The dataset might be empty or num_examples is 0.")
        # This check was inside the original if, makes more sense here or after loop
        # if total_examples == 0: # This condition will always be true here
        #      logger.info("\nNo examples were processed that had extractable ground truth answers.")
        return

    logger.info(f"Preparing {num_to_run} examples for concurrent execution with concurrency limit {args.concurrency}...")
    semaphore = asyncio.Semaphore(args.concurrency)

    async def solve_with_semaphore(question_data: str, answer_data: str) -> GSM8KResult:
        async with semaphore:
            return await solver.solve(question_data, true_answer_full=answer_data)

    for i in range(num_to_run):
        example = dataset[i]
        question = example["question"]
        true_answer_full = example["answer"]
        tasks.append(solve_with_semaphore(question, true_answer_full))

    logger.info(f"Running benchmark on {len(tasks)} examples concurrently (limit: {args.concurrency})...")
    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Benchmarking Progress"):
        result: GSM8KResult = await future

        if result.extracted_true_answer is None:
            # Using result.question as example index (i) is not directly available with as_completed
            logger.warning(
                f"Could not extract ground truth answer for question (Q: {result.question[:50]}...). Skipping."
            )
            logger.debug(f"  Skipped Question: {result.question}")
            logger.debug(f"  Full Ground Truth for Skipped: {result.true_answer_full}")
            continue # Skip this result from accuracy calculation

        total_examples +=1 # Only count examples where true answer could be extracted

        if result.extracted_model_answer is not None and \
           result.extracted_model_answer == result.extracted_true_answer:
            correct_answers += 1

    if total_examples > 0:
        accuracy = (correct_answers / total_examples) * 100
        logger.info("\n--- Benchmark Summary ---")
        logger.info(f"Total examples processed (with extractable true answers): {total_examples}")
        logger.info(f"Correct answers: {correct_answers}")
        logger.info(f"Accuracy: {accuracy:.2f}%")
    else:
        logger.info("\nNo examples were processed that had extractable ground truth answers.")


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
        dataset = GSM8KDataset(split=args.data_split, config=args.data_config, shuffle=args.shuffle)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    solver: GSM8KSolver

    if args.no_cache:
        logger.info("Caching is disabled by --no-cache flag.")
        logger.info(f"Initializing GSM8KSolver with model: {args.model_name} (caching disabled)...")
        solver = GSM8KSolver(
            model_name=args.model_name,
            prompt_template=prompt_template_content,
            cache_manager=None,
            prompt_template_name=prompt_template_name,
            data_split=None,
            data_config=None,
            run_id=None
        )
        asyncio.run(run_benchmarks(args, logger, dataset, solver)) # Changed to asyncio.run
    else:
        cache_db_path = "llm_benchmarks_cache.sqlite3"
        logger.info(f"Cache database path: {cache_db_path}")
        logger.info(f"Cache usage enabled (explicitly not using --no-cache).")

        with CacheManager(db_path=cache_db_path, use_cache=True) as cache_manager:
            cache_manager.init_db()

            logger.info(f"Initializing GSM8KSolver with model: {args.model_name} (caching enabled)...")
            solver = GSM8KSolver(
                model_name=args.model_name,
                prompt_template=prompt_template_content,
                cache_manager=cache_manager,
                prompt_template_name=prompt_template_name,
                data_split=args.data_split,
                data_config=args.data_config,
                run_id=run_id
            )
            asyncio.run(run_benchmarks(args, logger, dataset, solver)) # Changed to asyncio.run


if __name__ == "__main__":
    main()
