import logging
import re
from llm_benchmarks.model.model import OpenRouterPrompt

logger = logging.getLogger(__name__)


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


GSM8K_ANSWER_REGEX = r"#### (\d+\.?\d*)"

def extract_gsm8k_answer(answer_str: str) -> str | None:
    match = re.search(GSM8K_ANSWER_REGEX, answer_str)
    if match:
        return match.group(1)
    return None

class GSM8KSolver:
    def __init__(self, model_name: str, prompt_template: str):
        self.model = OpenRouterPrompt(prompt=prompt_template, model=model_name)
        self.prompt_template = prompt_template

    def solve(self, question: str, true_answer_full: str) -> GSM8KResult:
        model_response = self.model.execute_prompt(content=question)

        match_model = re.search(r"The final answer is \$?([\d,]*\.?\d+)", model_response)
        if match_model:
            extracted_model_answer = match_model.group(1).replace(',', '')
            if extracted_model_answer.endswith(".00"):
                extracted_model_answer = extracted_model_answer[:-3]
            elif extracted_model_answer.endswith(".0"):
                extracted_model_answer = extracted_model_answer[:-2]
        else:
            extracted_model_answer = None

        extracted_true_answer = extract_gsm8k_answer(true_answer_full)

        logger.debug(f"Question: {question}")
        logger.debug(f"Ground Truth Answer (Full): {true_answer_full}")
        logger.debug(f"Ground Truth Answer (Extracted): {extracted_true_answer}")
        logger.debug(f"Model's Full Response: {model_response}")
        logger.debug(f"Model's Predicted Answer (Extracted): {extracted_model_answer}")

        is_correct = (
            extracted_model_answer is not None
            and extracted_true_answer is not None
            and extracted_model_answer == extracted_true_answer
        )
        logger.debug(f"Result: {'CORRECT' if is_correct else 'INCORRECT'}")

        return GSM8KResult(
            question=question,
            model_response=model_response,
            extracted_model_answer=extracted_model_answer,
            true_answer_full=true_answer_full,
            extracted_true_answer=extracted_true_answer,
        )

    def __call__(self, question: str, true_answer_full: str) -> GSM8KResult:
        return self.solve(question, true_answer_full=true_answer_full)
