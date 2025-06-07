import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

class OpenRouterPrompt:
    """
    Class to interact with OpenRouter API using a prompt template.
    The prompt must contain a '{content}' placeholder.
    """
    def __init__(self, prompt: str, model: str = "deepseek/deepseek-r1-0528:free"):
        if '{content}' not in prompt:
            raise ValueError("Prompt must contain a '{content}' placeholder.")
        self.prompt = prompt
        self.model = model
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )

    def execute_prompt(self, content: str) -> str:
        """
        Executes the prompt with the given content.
        """
        try:
            full_prompt = self.prompt.format(content=content)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"An error occurred with model {self.model}: {e}")
            return f"Error: Could not get a response. Details: {e}"

    def __call__(self, content: str) -> str:
        return self.execute_prompt(content)