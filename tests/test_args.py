import pytest
from unittest import mock
import sys

# Assuming parse_arguments is in src.llm_benchmarks.utils.args
from src.llm_benchmarks.utils.args import parse_arguments

# Mock prompt contents
DEFAULT_PROMPT_CONTENT = "This is the default prompt: {content}"
RIGOROUS_PROMPT_CONTENT = "This is the rigorous prompt: {content}"

@mock.patch('src.llm_benchmarks.utils.args.load_prompt')
@mock.patch('src.llm_benchmarks.utils.args.get_available_prompts')
def test_load_default_prompt(mock_get_available_prompts, mock_load_prompt):
    """Tests loading of the default prompt when no keyword is specified."""
    mock_get_available_prompts.return_value = ['default', 'rigorous']
    # Configure mock_load_prompt to return DEFAULT_PROMPT_CONTENT when 'default' is passed
    mock_load_prompt.side_effect = lambda keyword: DEFAULT_PROMPT_CONTENT if keyword == 'default' else None

    with mock.patch('sys.argv', ['script_name']):
        args = parse_arguments()
        assert args.prompt == 'default' # Check that the keyword defaults correctly
        assert args.prompt_template == DEFAULT_PROMPT_CONTENT
        mock_load_prompt.assert_called_once_with('default')
        mock_get_available_prompts.assert_called_once()

@mock.patch('src.llm_benchmarks.utils.args.load_prompt')
@mock.patch('src.llm_benchmarks.utils.args.get_available_prompts')
def test_load_specific_prompt_rigorous(mock_get_available_prompts, mock_load_prompt):
    """Tests loading of a specific prompt ('rigorous')."""
    mock_get_available_prompts.return_value = ['default', 'rigorous']
    # Configure mock_load_prompt for 'rigorous'
    mock_load_prompt.side_effect = lambda keyword: RIGOROUS_PROMPT_CONTENT if keyword == 'rigorous' else None

    with mock.patch('sys.argv', ['script_name', '--prompt', 'rigorous']):
        args = parse_arguments()
        assert args.prompt == 'rigorous'
        assert args.prompt_template == RIGOROUS_PROMPT_CONTENT
        mock_load_prompt.assert_called_once_with('rigorous')
        mock_get_available_prompts.assert_called_once()

@mock.patch('src.llm_benchmarks.utils.args.get_available_prompts')
def test_load_invalid_choice_prompt(mock_get_available_prompts, capsys):
    """Tests behavior when an invalid prompt keyword (not in choices) is used."""
    mock_get_available_prompts.return_value = ['default', 'rigorous']

    with mock.patch('sys.argv', ['script_name', '--prompt', 'non_existent_prompt']):
        with pytest.raises(SystemExit) as e:
            parse_arguments()

        assert e.type == SystemExit
        assert e.value.code == 2 # Argparse error exit code

        captured = capsys.readouterr()
        # Check that argparse's error message for invalid choice is in stderr
        assert "invalid choice: 'non_existent_prompt'" in captured.err
        assert "(choose from 'default', 'rigorous')" in captured.err
        mock_get_available_prompts.assert_called_once()

@mock.patch('src.llm_benchmarks.utils.args.load_prompt')
@mock.patch('src.llm_benchmarks.utils.args.get_available_prompts')
def test_load_prompt_file_not_found_after_parse(mock_get_available_prompts, mock_load_prompt, capsys):
    """
    Tests behavior if a chosen prompt (valid choice) file is not found by load_prompt.
    This simulates a race condition or an issue where a file listed by get_available_prompts
    is then not found by load_prompt.
    """
    SELECTED_PROMPT = 'default'
    AVAILABLE_PROMPTS = [SELECTED_PROMPT, 'rigorous']
    mock_get_available_prompts.return_value = AVAILABLE_PROMPTS
    # Simulate load_prompt raising FileNotFoundError for the selected prompt
    mock_load_prompt.side_effect = FileNotFoundError(f"Mocked FileNotFoundError for {SELECTED_PROMPT}.txt")

    with mock.patch('sys.argv', ['script_name', '--prompt', SELECTED_PROMPT]):
        with pytest.raises(SystemExit) as e:
            parse_arguments()

        assert e.type == SystemExit
        assert e.value.code == 2 # argparse.ArgumentParser.error exits with 2

        captured = capsys.readouterr()
        # Check that the custom error message from parse_arguments (calling parser.error) is in stderr
        assert f"Selected prompt file for keyword '{SELECTED_PROMPT}' not found." in captured.err
        # Corrected assertion for available prompts message
        assert f"Available prompts: {', '.join(AVAILABLE_PROMPTS)}" in captured.err
        # get_available_prompts is called once for choices, and again inside the error handler if load_prompt fails.
        assert mock_get_available_prompts.call_count == 2
        mock_load_prompt.assert_called_once_with(SELECTED_PROMPT)


@mock.patch('src.llm_benchmarks.utils.args.load_prompt')
@mock.patch('src.llm_benchmarks.utils.args.get_available_prompts')
def test_parse_arguments_num_threads(mock_get_available_prompts, mock_load_prompt):
    """Tests parsing of the --num-threads argument."""
    # Mock basic prompt loading functionality to satisfy parser
    mock_get_available_prompts.return_value = ['default']
    mock_load_prompt.return_value = "prompt_content_for_default"

    # Test case 1: --num-threads is provided
    with mock.patch('sys.argv', ['script_name', '--num-threads', '4', '--prompt', 'default']):
        args = parse_arguments()
        assert args.num_threads == 4
        assert args.prompt == 'default' # Ensure other args are fine

    # Reset mocks for the next case if necessary (though parse_arguments re-evaluates them)
    mock_load_prompt.reset_mock()
    mock_get_available_prompts.reset_mock() # Reset for call count checks if any new ones are added
    mock_get_available_prompts.return_value = ['default'] # Re-assign return value
    mock_load_prompt.return_value = "prompt_content_for_default"


    # Test case 2: --num-threads is not provided (should default to 1)
    with mock.patch('sys.argv', ['script_name', '--prompt', 'default']):
        args = parse_arguments()
        assert args.num_threads == 1
        assert args.prompt == 'default'

    # Test case 3: --num-threads with a different value
    mock_load_prompt.reset_mock()
    mock_get_available_prompts.reset_mock()
    mock_get_available_prompts.return_value = ['default']
    mock_load_prompt.return_value = "prompt_content_for_default"

    with mock.patch('sys.argv', ['script_name', '--num-threads', '10', '--model-name', 'test_model', '--prompt', 'default']):
        args = parse_arguments()
        assert args.num_threads == 10
        assert args.model_name == 'test_model'
        assert args.prompt == 'default'
