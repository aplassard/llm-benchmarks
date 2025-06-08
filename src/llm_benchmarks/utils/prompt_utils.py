import os

PROMPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "prompts"))

def get_prompt_path(keyword: str) -> str:
    """Returns the full path to the prompt file for a given keyword."""
    return os.path.join(PROMPT_DIR, f"{keyword}.txt")

def load_prompt(keyword: str) -> str:
    """Loads and returns the content of the prompt file for a given keyword."""
    prompt_file_path = get_prompt_path(keyword)
    if not os.path.exists(prompt_file_path):
        raise FileNotFoundError(f"Prompt file not found: {prompt_file_path}")
    with open(prompt_file_path, "r", encoding="utf-8") as f:
        return f.read()

def get_available_prompts() -> list[str]:
    """Returns a list of available prompt keywords."""
    if not os.path.isdir(PROMPT_DIR):
        return []
    keywords = [
        f[:-4]  # Remove .txt extension
        for f in os.listdir(PROMPT_DIR)
        if f.endswith(".txt") and os.path.isfile(os.path.join(PROMPT_DIR, f))
    ]
    return sorted(keywords)
