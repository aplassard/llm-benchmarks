import pytest
import os
from unittest import mock # mock is still used for other things if needed, but not for PROMPT_DIR here.

# Functions to test
from src.llm_benchmarks.utils.prompt_utils import (
    get_prompt_path,
    load_prompt,
    get_available_prompts,
    PROMPT_DIR as ACTUAL_PROMPT_DIR # Import the original for one test
)

# Define mock prompt contents
MOCK_DEFAULT_CONTENT = "Mock default prompt content"
MOCK_RIGOROUS_CONTENT = "Mock rigorous prompt content"
MOCK_CHAIN_OF_THOUGHT_CONTENT = """Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?

Natalia sold 48 clips in April.
In May, she sold half as many clips as in April, so she sold 48 / 2 = 24 clips.
Altogether, she sold 48 + 24 = 72 clips.
The final answer is 72

Question: {content}

Please think step by step and show your work before arriving at the solution. End your response with the phrase 'The final answer is X' where X is the numerical answer."""
MOCK_LLAMA3_2_WEAKNESS_ADDRESSING_CONTENT = "Mock Llama 3.2 weakness addressing prompt"
MOCK_SIMPLE_CONTENT = """Question: {content}

Ensure your response ends with the phrase 'The final answer is X' where X is the numerical answer."""

@pytest.fixture
def temp_prompt_files(tmp_path):
    prompts_test_dir = tmp_path / "prompts_test_for_monkeypatch" # Use a distinct name for clarity
    prompts_test_dir.mkdir()

    (prompts_test_dir / "default.txt").write_text(MOCK_DEFAULT_CONTENT)
    (prompts_test_dir / "rigorous.txt").write_text(MOCK_RIGOROUS_CONTENT)
    (prompts_test_dir / "chain_of_thought.txt").write_text(MOCK_CHAIN_OF_THOUGHT_CONTENT)
    (prompts_test_dir / "llama3_2_weakness_addressing.txt").write_text(MOCK_LLAMA3_2_WEAKNESS_ADDRESSING_CONTENT)
    (prompts_test_dir / "simple.txt").write_text(MOCK_SIMPLE_CONTENT)
    (prompts_test_dir / "another.txt").write_text("Another prompt")
    (prompts_test_dir / "not_a_prompt.md").write_text("This is not a prompt file")

    return prompts_test_dir

def test_get_prompt_path_uses_actual_prompt_dir():
    # This test ensures get_prompt_path uses the PROMPT_DIR constant from the module.
    # It does not modify PROMPT_DIR.
    keyword = "test_prompt"
    # ACTUAL_PROMPT_DIR is the original, unpatched PROMPT_DIR
    expected_path = os.path.join(ACTUAL_PROMPT_DIR, f"{keyword}.txt")
    assert get_prompt_path(keyword) == expected_path

def test_load_prompt_success(temp_prompt_files, monkeypatch):
    """Tests successfully loading an existing prompt using monkeypatch."""
    monkeypatch.setattr('src.llm_benchmarks.utils.prompt_utils.PROMPT_DIR', str(temp_prompt_files))

    content = load_prompt("default")
    assert content == MOCK_DEFAULT_CONTENT

    content_rigorous = load_prompt("rigorous")
    assert content_rigorous == MOCK_RIGOROUS_CONTENT

    content_cot = load_prompt("chain_of_thought")
    assert content_cot == MOCK_CHAIN_OF_THOUGHT_CONTENT

    content_llama_weakness = load_prompt("llama3_2_weakness_addressing")
    assert content_llama_weakness == MOCK_LLAMA3_2_WEAKNESS_ADDRESSING_CONTENT

    content_simple = load_prompt("simple")
    assert content_simple == MOCK_SIMPLE_CONTENT

def test_load_prompt_file_not_found(temp_prompt_files, monkeypatch):
    """Tests FileNotFoundError for a non-existent prompt with monkeypatch."""
    monkeypatch.setattr('src.llm_benchmarks.utils.prompt_utils.PROMPT_DIR', str(temp_prompt_files))

    with pytest.raises(FileNotFoundError) as excinfo:
        load_prompt("non_existent_prompt")
    assert "Prompt file not found" in str(excinfo.value)
    assert "non_existent_prompt.txt" in str(excinfo.value)

def test_get_available_prompts(temp_prompt_files, monkeypatch):
    """Tests listing available prompt keywords with monkeypatch."""
    monkeypatch.setattr('src.llm_benchmarks.utils.prompt_utils.PROMPT_DIR', str(temp_prompt_files))

    available = get_available_prompts()
    assert sorted(available) == sorted([
        "another",
        "default",
        "rigorous",
        "chain_of_thought",
        "llama3_2_weakness_addressing",
        "simple"
    ])

def test_get_available_prompts_empty(tmp_path, monkeypatch):
    """Tests listing when the directory is empty or has no .txt files, with monkeypatch."""
    empty_prompts_dir = tmp_path / "empty_prompts_for_monkeypatch"
    empty_prompts_dir.mkdir()
    (empty_prompts_dir / "readme.md").write_text("notes")
    monkeypatch.setattr('src.llm_benchmarks.utils.prompt_utils.PROMPT_DIR', str(empty_prompts_dir))

    available = get_available_prompts()
    assert available == []

def test_get_available_prompts_no_directory(tmp_path, monkeypatch):
    """Tests listing when the prompt directory does not exist, with monkeypatch."""
    non_existent_dir = tmp_path / "non_existent_prompts_dir_for_monkeypatch"
    # Do not create this directory
    monkeypatch.setattr('src.llm_benchmarks.utils.prompt_utils.PROMPT_DIR', str(non_existent_dir))

    available = get_available_prompts()
    assert available == []
