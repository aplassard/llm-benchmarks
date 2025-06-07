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
        # that follows "The final answer is " and stops at a word boundary.
        match = re.search(r"The final answer is (\d+\.?\d*)\b", response)
        if match:
            return match.group(1)
        return None

    def __call__(self, question: str) -> str | None:
        return self.solve(question)
