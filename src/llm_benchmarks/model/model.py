import os
import time
import random
from openai import OpenAI, APIStatusError, APITimeoutError, APIConnectionError, RateLimitError
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

    # Retry parameters
    MAX_RETRIES = 5
    INITIAL_DELAY = 1.0  # seconds
    BACKOFF_FACTOR = 2.0
    JITTER_FACTOR = 0.1 # Max 10% jitter

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
        Executes the prompt with the given content, with exponential backoff.
        Returns the full ChatCompletion object or None if an error occurs after retries.
        """
        full_prompt = self.prompt.format(content=content)
        current_delay = self.INITIAL_DELAY

        for attempt in range(self.MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": full_prompt}],
                    temperature=0, # Assuming temperature 0 for benchmarks
                )
                return response
            except (APITimeoutError, APIConnectionError, RateLimitError) as e:
                logger.warning(
                    f"Retryable API error for model {self.model} (attempt {attempt + 1}/{self.MAX_RETRIES}): {type(e).__name__} - {e}. Retrying..."
                )
            except APIStatusError as e:
                # Retry on specific HTTP status codes
                if e.status_code in [429, 500, 502, 503, 504]:
                    logger.warning(
                        f"Retryable API StatusError for model {self.model} (attempt {attempt + 1}/{self.MAX_RETRIES}): Status {e.status_code} - {e}. Retrying..."
                    )
                else:
                    logger.error(
                        f"Non-retryable APIStatusError for model {self.model}: Status {e.status_code} - {e}. No retries."
                    )
                    return None # Non-retryable status code
            except Exception as e: # Catch any other unexpected errors
                logger.error(
                    f"An unexpected error occurred with model {self.model} during API call (attempt {attempt + 1}/{self.MAX_RETRIES}): {type(e).__name__} - {e}"
                )
                # Depending on policy, you might not want to retry on *all* generic Exceptions.
                # For now, we treat unexpected errors as potentially retryable to be robust,
                # but this could be refined. If it's something like AuthenticationError, retrying won't help.
                # However, the specific OpenAI errors should cover most common retryable scenarios.
                # If this is the last attempt, it will fall through to the final error log.
                if attempt == self.MAX_RETRIES - 1:
                    logger.error(f"Final attempt failed for model {self.model} due to unexpected error: {e}")
                    return None

            if attempt < self.MAX_RETRIES - 1:
                # Calculate delay with jitter
                jitter = random.uniform(0, current_delay * self.JITTER_FACTOR)
                sleep_time = current_delay + jitter
                logger.info(f"Waiting {sleep_time:.2f}s before next retry for model {self.model}...")
                time.sleep(sleep_time)
                current_delay *= self.BACKOFF_FACTOR
            else:
                logger.error(
                    f"API call failed for model {self.model} after {self.MAX_RETRIES} retries."
                )
                return None

        return None # Should be unreachable if loop completes, but as a fallback.

    def __call__(self, content: str) -> ChatCompletion | None:
        return self.execute_prompt(content)
