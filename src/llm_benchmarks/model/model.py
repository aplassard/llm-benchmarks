import os
from openai import AsyncOpenAI, APIConnectionError, RateLimitError # Changed OpenAI to AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from dotenv import load_dotenv
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import asyncio # Added asyncio

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
        self.client = AsyncOpenAI( # Changed OpenAI to AsyncOpenAI
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )

    @retry(
        retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60)
    )
    async def execute_prompt(self, content: str) -> ChatCompletion | None: # Changed to async def
        """
        Executes the prompt with the given content.
        Returns the full ChatCompletion object or None if an error occurs.
        """
        try:
            full_prompt = self.prompt.format(content=content)
            response = await self.client.chat.completions.create( # Added await
                model=self.model,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0,
            )
            return response
        except Exception as e:
            logger.error(f"An error occurred with model {self.model} during API call after multiple retries: {e}")
            raise  # Re-raise the exception after logging

    async def __call__(self, content: str) -> ChatCompletion | None: # Changed to async def
        return await self.execute_prompt(content) # Added await
