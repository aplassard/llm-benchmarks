import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

def get_openrouter_response(prompt: str, model_name: str) -> str:
    """
    Sends a prompt to the OpenRouter API and gets a response
    from a specified model.

    Args:
        prompt: The text prompt to send to the model.
        model_name: The identifier for the model to use
                    (e.g., "google/gemini-pro", "mistralai/mistral-7b-instruct").

    Returns:
        The model's response as a string.
    """
    try:
        # Point the client to the OpenRouter API
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )

        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,  # Keep it deterministic for benchmarks
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"An error occurred with model {model_name}: {e}")
        return f"Error: Could not get a response. Details: {e}"