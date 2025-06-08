import pytest
import os
from unittest import mock

# Functions to test from src.llm_benchmarks.utils.prompts
from src.llm_benchmarks.utils.prompts import (
    get_prompt_path,
    load_prompt,
    get_available_prompts,
    PROMPT_DIR,  # Import for patching its value or for reference
)

# Define mock prompt contents
MOCK_DEFAULT_CONTENT = "Mock default prompt content"
MOCK_RIGOROUS_CONTENT = "Mock rigorous prompt content"

@pytest.fixture
def temp_prompt_files(tmp_path):
    # Create a temporary 'prompts' directory structure similar to the real one
    # but within pytest's tmp_path for this test session.
    # tmp_path is a Pathlib object.
    prompts_test_dir = tmp_path / "prompts_test"
    prompts_test_dir.mkdir()

    # Create some mock prompt files
    (prompts_test_dir / "default.txt").write_text(MOCK_DEFAULT_CONTENT)
    (prompts_test_dir / "rigorous.txt").write_text(MOCK_RIGOROUS_CONTENT)
    (prompts_test_dir / "another.txt").write_text("Another prompt")
    (prompts_test_dir / "not_a_prompt.md").write_text("This is not a prompt file")

    return prompts_test_dir

def test_get_prompt_path():
    """Tests the get_prompt_path function for correct path construction."""
    keyword = "test_prompt"
    expected_path = os.path.join(PROMPT_DIR, f"{keyword}.txt")
    assert get_prompt_path(keyword) == expected_path

# Patch PROMPT_DIR in the prompts module to use our temp_prompt_files fixture
@mock.patch('src.llm_benchmarks.utils.prompts.PROMPT_DIR', new_callable=mock.PropertyMock)
def test_load_prompt_success(mock_prompt_dir_prop, temp_prompt_files):
    """Tests successfully loading an existing prompt."""
    mock_prompt_dir_prop.return_value = str(temp_prompt_files) # PROMPT_DIR will now point to temp_prompt_files

    content = load_prompt("default")
    assert content == MOCK_DEFAULT_CONTENT

    content_rigorous = load_prompt("rigorous")
    assert content_rigorous == MOCK_RIGOROUS_CONTENT

@mock.patch('src.llm_benchmarks.utils.prompts.PROMPT_DIR', new_callable=mock.PropertyMock)
def test_load_prompt_file_not_found(mock_prompt_dir_prop, temp_prompt_files):
    """Tests that FileNotFoundError is raised for a non-existent prompt keyword."""
    mock_prompt_dir_prop.return_value = str(temp_prompt_files)

    with pytest.raises(FileNotFoundError) as excinfo:
        load_prompt("non_existent_prompt")
    assert "Prompt file not found" in str(excinfo.value)
    assert "non_existent_prompt.txt" in str(excinfo.value)

@mock.patch('src.llm_benchmarks.utils.prompts.PROMPT_DIR', new_callable=mock.PropertyMock)
def test_get_available_prompts(mock_prompt_dir_prop, temp_prompt_files):
    """Tests listing available prompt keywords."""
    mock_prompt_dir_prop.return_value = str(temp_prompt_files)

    available = get_available_prompts()
    assert sorted(available) == sorted(["another", "default", "rigorous"])

@mock.patch('src.llm_benchmarks.utils.prompts.PROMPT_DIR', new_callable=mock.PropertyMock)
def test_get_available_prompts_empty(mock_prompt_dir_prop, tmp_path):
    """Tests listing available prompts when the directory is empty or contains no .txt files."""
    empty_prompts_dir = tmp_path / "empty_prompts"
    empty_prompts_dir.mkdir()
    (empty_prompts_dir / "readme.md").write_text("notes")
    mock_prompt_dir_prop.return_value = str(empty_prompts_dir)

    available = get_available_prompts()
    assert available == []

@mock.patch('src.llm_benchmarks.utils.prompts.PROMPT_DIR', new_callable=mock.PropertyMock)
def test_get_available_prompts_no_directory(mock_prompt_dir_prop, tmp_path):
    """Tests listing available prompts when the prompt directory does not exist."""
    non_existent_dir = tmp_path / "non_existent_prompts_dir"
    # Do not create this directory
    mock_prompt_dir_prop.return_value = str(non_existent_dir)

    available = get_available_prompts()
    assert available == []
