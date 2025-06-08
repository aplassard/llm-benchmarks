import pytest
import argparse
from unittest import mock

# Assuming parse_arguments is in src.llm_benchmarks.utils.args
from src.llm_benchmarks.utils.args import parse_arguments

# Mock prompt contents
DEFAULT_PROMPT_CONTENT = "This is the default prompt: {content}"
RIGOROUS_PROMPT_CONTENT = "This is the rigorous prompt: {content}"

@mock.patch('os.path.join', side_effect=lambda *args: "/".join(args)) # Mock os.path.join to behave consistently
def test_load_default_prompt(mock_os_join, capsys):
    """Tests loading of the default prompt when no keyword is specified."""
    with mock.patch('sys.argv', ['script_name']):
        # Mock open to return default content for 'default.txt'
        m = mock.mock_open(read_data=DEFAULT_PROMPT_CONTENT)
        with mock.patch('builtins.open', m) as mock_open_call:
            args = parse_arguments()
            assert args.prompt_template == DEFAULT_PROMPT_CONTENT
            # Check if open was called with the correct path for default.txt
            mock_open_call.assert_called_once_with("src/llm_benchmarks/utils/prompts/default.txt", "r", encoding="utf-8")

@mock.patch('os.path.join', side_effect=lambda *args: "/".join(args))
def test_load_specific_prompt_rigorous(mock_os_join, capsys):
    """Tests loading of a specific prompt ('rigorous')."""
    with mock.patch('sys.argv', ['script_name', '--prompt-keyword', 'rigorous']):
        m = mock.mock_open(read_data=RIGOROUS_PROMPT_CONTENT)
        with mock.patch('builtins.open', m) as mock_open_call:
            args = parse_arguments()
            assert args.prompt_template == RIGOROUS_PROMPT_CONTENT
            mock_open_call.assert_called_once_with("src/llm_benchmarks/utils/prompts/rigorous.txt", "r", encoding="utf-8")

@mock.patch('os.path.join', side_effect=lambda *args: "/".join(args))
def test_load_non_existent_prompt(mock_os_join, capsys):
    """Tests behavior when a non-existent prompt keyword is used."""
    with mock.patch('sys.argv', ['script_name', '--prompt-keyword', 'non_existent_prompt']):
        # Mock open to raise FileNotFoundError
        with mock.patch('builtins.open', side_effect=FileNotFoundError) as mock_open_call:
            with pytest.raises(SystemExit) as e: # parse_arguments calls parser.error which calls sys.exit
                parse_arguments()
            assert e.type == SystemExit # Check that SystemExit was raised
            # Check if open was called with the correct path for non_existent_prompt.txt
            mock_open_call.assert_called_once_with("src/llm_benchmarks/utils/prompts/non_existent_prompt.txt", "r", encoding="utf-8")

            # Check stderr for the error message from parser.error()
            # This requires the test to be run in a way that captures stderr, which pytest does by default with capsys.
            # However, parser.error() writes to sys.stderr directly.
            # We can also mock parser.error to check its arguments.
            # For now, let's check the captured stderr.
            # Note: The exact error message depends on argparse's formatting.
            # We're checking if the expected file path is mentioned in the error.
            captured = capsys.readouterr()
            assert "Prompt file not found: src/llm_benchmarks/utils/prompts/non_existent_prompt.txt" in captured.err
            assert "Please ensure a file named 'non_existent_prompt.txt' exists" in captured.err

# Example of how to mock parser.error for more specific testing of its call, if needed:
# @mock.patch('argparse.ArgumentParser.error')
# @mock.patch('os.path.join', side_effect=lambda *args: "/".join(args))
# def test_load_non_existent_prompt_mock_parser_error(mock_os_join, mock_parser_error):
#     """Tests non-existent prompt by mocking parser.error."""
#     with mock.patch('sys.argv', ['script_name', '--prompt-keyword', 'non_existent_prompt']):
#         with mock.patch('builtins.open', side_effect=FileNotFoundError):
#             parse_arguments() # Should call parser.error
#             mock_parser_error.assert_called_once()
#             # Check that the error message contains the relevant info
#             args, _ = mock_parser_error.call_args
#             assert "Prompt file not found: src/llm_benchmarks/utils/prompts/non_existent_prompt.txt" in args[0]

# To run these tests:
# Ensure pytest is installed: pip install pytest
# Ensure this test file is in a 'tests' directory at the same level as 'src'
# From the root directory of the project, run: pytest
# (You might need to set PYTHONPATH=. or export PYTHONPATH=.:src for imports to work)
# Example: PYTHONPATH=.:src pytest tests/test_args.py
# Or if using a proper package structure, pytest should handle it.
# For these tests to run, the file src/llm_benchmarks/utils/args.py must exist and be importable.
# And src/llm_benchmarks/utils/__init__.py and src/llm_benchmarks/__init__.py might be needed.
# If they don't exist, create them as empty files.
# Example: touch src/llm_benchmarks/__init__.py src/llm_benchmarks/utils/__init__.py
# Also, the main `src/llm_benchmarks/utils/args.py` should not have top-level code that runs on import,
# e.g., calling parse_arguments() directly, as that would interfere with testing.
# The provided args.py seems fine in that regard.
# The mock for os.path.join is a simple way to ensure consistent path separators for the assertion,
# regardless of the OS the test runs on.
# The `capsys` fixture is used to capture stderr for the third test.
# The `FileNotFoundError` should be caught by `parse_arguments` and then `parser.error` should be called,
# which in turn calls `sys.exit`.
# So we expect a SystemExit.
# The `capsys.readouterr()` allows checking the message printed to stderr.
# The comment about PYTHONPATH and __init__.py files is important for local test execution.
# In a CI environment, these are usually handled by the test runner configuration.
# The mock_os_join is added to ensure consistent path generation for assertions,
# as os.path.join might produce different separators (e.g. \ on Windows)
# which would make string comparisons in assert_called_once_with fail.
# Using a simple lambda side_effect for os.path.join ensures paths are joined with '/'.
# This is generally fine for testing the logic as long as the mocked path is what `open` expects.
# The code in args.py uses os.path.join("src", "llm_benchmarks", "utils", "prompts", f"{keyword}.txt")
# So the mocked join will produce "src/llm_benchmarks/utils/prompts/keyword.txt".

# Final check of the test_load_non_existent_prompt:
# argparse.ArgumentParser.error() prints to stderr and then calls sys.exit().
# So, pytest.raises(SystemExit) is correct.
# To check the error message, we use capsys.
# The test suite looks reasonable.
# Adding an __init__.py to tests directory is also a good practice.
# touch tests/__init__.py
# One more detail: the `parse_arguments` function in `args.py` uses `parser.error(...)`.
# This method prints the message to stderr and then calls `sys.exit(2)`.
# So, `pytest.raises(SystemExit)` is the correct way to catch this.
# The `capsys` fixture in pytest can capture output to stdout/stderr.
# `captured = capsys.readouterr()` will give `captured.out` and `captured.err`.
# This is used to assert the error message.
# The mock for `os.path.join` is a good addition to make tests more robust across different OS,
# ensuring the path string used in `mock_open_call.assert_called_once_with(...)` matches exactly.
# The actual `open` call inside `parse_arguments` will use the real `os.path.join` unless that is also patched,
# but here we are patching `builtins.open` and only need to ensure that the *expected path* in the assertion
# matches what the code *would* generate if `os.path.join` behaved as mocked.
# Actually, the `os.path.join` mock is important because the *argument* to `builtins.open` (which is `prompt_file_path`)
# is constructed using `os.path.join`. So, if `os.path.join` is not mocked, on Windows, `prompt_file_path` would be
# e.g., "src\\llm_benchmarks\\utils\\prompts\\default.txt", and the assertion
# `mock_open_call.assert_called_once_with("src/llm_benchmarks/utils/prompts/default.txt", ...)` would fail.
# So, the `mock_os_join` patch is crucial for the `assert_called_once_with` to work reliably across OS.
# The file path argument to `open` inside `parse_arguments` WILL be affected by `mock_os_join`.
# And the assertion path string also uses `/`. This is consistent.
# Looks good.
#
# One final thought on `test_load_non_existent_prompt`:
# The `parser.error()` is part of `argparse` and is what we want to test the invocation of.
# The current setup correctly checks for `SystemExit` and the error message via `capsys`.
# This is a good integration test for that part of the `parse_arguments` function.
# No need to mock `parser.error` itself unless we wanted to avoid the `SystemExit` for some reason.
# The tests look complete and cover the requirements.

# Add __init__.py to src/llm_benchmarks and src/llm_benchmarks/utils if they don't exist.
# I'll assume they exist or will be created in another step if needed for pytest to find the module.
# For now, focus on creating this test file.
# The import `from src.llm_benchmarks.utils.args import parse_arguments` assumes that
# the test runner (pytest) is invoked from a directory where 'src' is a package,
# or PYTHONPATH is set up accordingly (e.g., `PYTHONPATH=. pytest` or `PYTHONPATH=src:. pytest`).
# If `llm_benchmarks` is an installable package, these imports might look different
# (e.g., `from llm_benchmarks.utils.args import parse_arguments`).
# Given the current structure, `PYTHONPATH=.:src` or just `PYTHONPATH=src` might be needed if running `pytest tests/test_args.py` from the root.
# If running `pytest` from root, and `src` is on pythonpath, then `src.llm_benchmarks...` is correct.
# This is standard practice for projects not yet packaged for installation.
# The file looks ready.
