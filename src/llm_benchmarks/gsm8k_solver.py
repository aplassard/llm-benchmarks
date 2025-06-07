import re
from llm_benchmarks.model.model import OpenRouterPrompt


class GSM8KSolver:
    def __init__(self, model_name: str, prompt_template: str):
        self.model = OpenRouterPrompt(prompt=prompt_template, model=model_name)
        self.prompt_template = prompt_template

    def solve(self, question: str) -> str | None:
        # Prompt the model
        response = self.model.execute_prompt(content=question)

        # Extract the answer using a regex
        # This regex looks for a number (integer or decimal)
        # that follows "The final answer is ".
        match = re.search(r"The final answer is (\d+\.?\d*)", response)
        if match:
            return match.group(1)
        else:
            # Fallback regex: look for any number in the response
            # if the preferred format is not found.
            fallback_match = re.search(r"(\d+\.?\d+)", response)
            if fallback_match:
                return fallback_match.group(1)
            else:
                return None

    def __call__(self, question: str) -> str | None:
        return self.solve(question)
