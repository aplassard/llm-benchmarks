import re
from llm_benchmarks.model.model import OpenRouterPrompt


# Definition of GSM8KResult (NEW)
class GSM8KResult:
    def __init__(
        self,
        question: str,
        model_response: str,
        extracted_model_answer: str | None,
        true_answer_full: str,
        extracted_true_answer: str | None,
    ):
        self.question = question
        self.model_response = model_response
        self.extracted_model_answer = extracted_model_answer
        self.true_answer_full = true_answer_full
        self.extracted_true_answer = extracted_true_answer

# Assume extract_gsm8k_answer is defined here or imported for now
# For example:
# from ..main import extract_gsm8k_answer, GSM8K_ANSWER_REGEX
# Or it will be moved here in a subsequent step. For this subtask, we will add a placeholder.

# Moved from main.py (NEW)
GSM8K_ANSWER_REGEX = r"#### (\d+\.?\d*)"

def extract_gsm8k_answer(answer_str: str) -> str | None: # Moved from main.py (NEW)
    match = re.search(GSM8K_ANSWER_REGEX, answer_str)
    if match:
        return match.group(1)
    return None

class GSM8KSolver:
    def __init__(self, model_name: str, prompt_template: str, verbose: bool = False): # MODIFIED
        self.model = OpenRouterPrompt(prompt=prompt_template, model=model_name)
        self.prompt_template = prompt_template
        self.verbose = verbose # NEW

    def solve(self, question: str, true_answer_full: str) -> GSM8KResult:
        # Prompt the model
        model_response = self.model.execute_prompt(content=question)

        # Extract the answer from model's response
        match_model = re.search(r"The final answer is \$?([\d,]*\.?\d+)", model_response)
        if match_model:
            extracted_model_answer = match_model.group(1).replace(',', '')
            if extracted_model_answer.endswith(".00"):
                extracted_model_answer = extracted_model_answer[:-3]
            elif extracted_model_answer.endswith(".0"):
                extracted_model_answer = extracted_model_answer[:-2]
        else:
            extracted_model_answer = None

        # Extract the true answer from the provided full string
        # This will use the actual moved/imported function in later steps
        extracted_true_answer = extract_gsm8k_answer(true_answer_full) # Using moved function

        # NEW: Verbose printing logic
        if self.verbose:
            print(f"Question: {question}")
            print(f"Ground Truth Answer (Full): {true_answer_full}")
            print(f"Ground Truth Answer (Extracted): {extracted_true_answer}")
            print(f"Model's Full Response: {model_response}")
            print(f"Model's Predicted Answer (Extracted): {extracted_model_answer}")

            # A prediction is correct if both extracted answers are not None and they match.
            is_correct = (
                extracted_model_answer is not None
                and extracted_true_answer is not None
                and extracted_model_answer == extracted_true_answer
            )
            print(f"Result: {'CORRECT' if is_correct else 'INCORRECT'}")

        return GSM8KResult(
            question=question,
            model_response=model_response,
            extracted_model_answer=extracted_model_answer,
            true_answer_full=true_answer_full,
            extracted_true_answer=extracted_true_answer,
        )

    def __call__(self, question: str, true_answer_full: str) -> GSM8KResult: # Signature remains the same
        # MODIFIED LINE:
        return self.solve(question, true_answer_full=true_answer_full)
