import os
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger(__name__)

class OpenRouterPrompt:
    """
    Class to interact with OpenRouter API using a prompt template.
    The prompt must contain a '{content}' placeholder.
    """

    def __init__(self, prompt: str, model: str = "deepseek/deepseek-r1-0528:free"):
        if "{content}" not in prompt:
            raise ValueError("Prompt must contain a '{content}' placeholder.")
        self.prompt = prompt
        self.model = model
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )

    def execute_prompt(self, content: str) -> ChatCompletion | None:
        """
        Executes the prompt with the given content.
        Returns the full ChatCompletion object or None if an error occurs.
        """
        try:
            full_prompt = self.prompt.format(content=content)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0,
            )
            return response
        except Exception as e:
            logger.error(f"An error occurred with model {self.model} during API call: {e}")
            return None

    def __call__(self, content: str) -> ChatCompletion | None:
        return self.execute_prompt(content)
