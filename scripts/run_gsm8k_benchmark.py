import argparse
import os
import re
from dotenv import load_dotenv

from llm_benchmarks.data.gsm8k import GSM8KDataset
from llm_benchmarks.gsm8k_solver import GSM8KSolver

# Load environment variables from .env file
load_dotenv()

# Regex to extract the final numerical answer from GSM8K ground truth
GSM8K_ANSWER_REGEX = r"#### (\d+\.?\d*)"

DEFAULT_PROMPT_TEMPLATE = """Question: {content}

Please provide a step-by-step solution and end your response with the phrase 'The final answer is X' where X is the numerical answer."""


def extract_gsm8k_answer(answer_str: str) -> str | None:
    match = re.search(GSM8K_ANSWER_REGEX, answer_str)
    if match:
        return match.group(1)
    return None


def main():
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
        print("Error: OPENROUTER_API_KEY environment variable not set.")
        print("Please set it in your .env file or environment.")
        return

    print(
        f"Loading GSM8K dataset (split: {args.data_split}, config: {args.data_config})..."
    )
    try:
        dataset = GSM8KDataset(split=args.data_split, config=args.data_config)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    print(f"Initializing GSM8KSolver with model: {args.model_name}...")
    solver = GSM8KSolver(
        model_name=args.model_name, prompt_template=args.prompt_template
    )

    correct_answers = 0
    total_examples = 0

    num_to_run = (
        len(dataset)
        if args.num_examples == -1
        else min(args.num_examples, len(dataset))
    )

    if num_to_run == 0:
        print("No examples to run. The dataset might be empty or num_examples is 0.")
        return

    print(f"Running benchmark on {num_to_run} examples...")

    for i in range(num_to_run):
        example = dataset[i]
        question = example["question"]
        true_answer_full = example["answer"]

        true_answer_extracted = extract_gsm8k_answer(true_answer_full)

        if true_answer_extracted is None:
            print(
                f"Warning: Could not extract ground truth answer for example {i+1}. Skipping."
            )
            if args.verbose:
                print(f"  Question: {question}")
                print(f"  Full Ground Truth: {true_answer_full}")
            continue

        total_examples += 1

        if args.verbose:
            print(f"\n--- Example {total_examples}/{num_to_run} ---")
            print(f"Question: {question}")
            print(f"Ground Truth Answer (Full): {true_answer_full}")
            print(f"Ground Truth Answer (Extracted): {true_answer_extracted}")

        model_answer = solver.solve(question)

        if args.verbose:
            print(f"Model's Predicted Answer (Extracted): {model_answer}")

        if model_answer == true_answer_extracted:
            correct_answers += 1
            if args.verbose:
                print("Result: CORRECT")
        else:
            if args.verbose:
                print("Result: INCORRECT")

        # Optional: Print progress periodically if not verbose
        if not args.verbose and (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{num_to_run} examples...")

    if total_examples > 0:
        accuracy = (correct_answers / total_examples) * 100
        print("\n--- Benchmark Summary ---")
        print(f"Total examples processed: {total_examples}")
        print(f"Correct answers: {correct_answers}")
        print(f"Accuracy: {accuracy:.2f}%")
    else:
        print("\nNo examples were processed that had extractable ground truth answers.")


if __name__ == "__main__":
    main()
