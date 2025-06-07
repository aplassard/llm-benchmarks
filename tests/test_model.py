# tests/test_model.py

import pytest
from unittest.mock import MagicMock
import httpx

# Import the specific exceptions from the openai library to simulate errors
from openai import AuthenticationError

# Import the class to be tested
# IMPORTANT: Adjust this import path to match your project structure
from llm_benchmarks.model import OpenRouterPrompt

# --- Test Group 1: Initialization (`__init__`) ---

def test_init_raises_error_on_bad_prompt():
    """
    Verifies that __init__ raises a ValueError if the prompt template
    is missing the required '{content}' placeholder.
    """
    with pytest.raises(ValueError, match="Prompt must contain a '{content}' placeholder."):
        OpenRouterPrompt(prompt="This is a bad prompt without the placeholder.")

def test_init_configures_client_correctly(mocker):
    """
    Verifies that __init__ initializes the OpenAI client with the correct
    OpenRouter-specific base_url and API key.
    """
    # Mock os.getenv to return a predictable fake API key
    mocker.patch('llm_benchmarks.model.model.os.getenv', return_value="fake-api-key")
    
    # Mock the entire OpenAI class to capture how it's instantiated
    mock_openai_class = mocker.patch('llm_benchmarks.model.model.OpenAI')

    # Now, initialize our class
    OpenRouterPrompt(prompt="This is a valid prompt with {content}.")

    # Assert that the OpenAI client was created with the right parameters
    mock_openai_class.assert_called_once_with(
        base_url="https://openrouter.ai/api/v1",
        api_key="fake-api-key"
    )

# --- Test Group 2: Execution Logic (`execute_prompt` and `__call__`) ---

@pytest.fixture
def mock_openai_client(mocker):
    """
    A fixture that creates a reusable mock of the OpenAI client instance.
    This mock will be injected into our OpenRouterPrompt class during tests.
    """
    # Create a mock for the client instance and its nested methods
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    # This is the fake content our successful tests will check for
    mock_response.choices[0].message.content = "This is a successful mock response."
    
    # Configure the mock's create method to return our fake response
    mock_client.chat.completions.create.return_value = mock_response

    # Patch the OpenAI class so that whenever it's called, it returns our mock_client
    mocker.patch('llm_benchmarks.model.model.OpenAI', return_value=mock_client)
    
    return mock_client

def test_execute_prompt_success(mock_openai_client):
    """
    Tests the "happy path" for execute_prompt: ensures the prompt is formatted
    correctly and the successful response is parsed and returned.
    """
    # Setup
    template = "Summarize this: {content}"
    model = "google/gemini-pro"
    prompt_instance = OpenRouterPrompt(prompt=template, model=model)

    # Action
    content_to_pass = "This is the text to summarize."
    result = prompt_instance.execute_prompt(content_to_pass)

    # Assertions
    # 1. Check that the returned result is correct
    assert result == "This is a successful mock response."

    # 2. Check that the API was called with the correctly formatted prompt
    expected_full_prompt = "Summarize this: This is the text to summarize."
    mock_openai_client.chat.completions.create.assert_called_once_with(
        model=model,
        messages=[{"role": "user", "content": expected_full_prompt}],
        temperature=0,
    )

def test_execute_prompt_handles_api_error(mock_openai_client):
    """
    Tests that the function handles a simulated API authentication error gracefully.
    """
    # 1. Create the minimal mock request and response objects the error needs
    mock_request = httpx.Request(method="POST", url="https://api.test.com")
    mock_response = httpx.Response(status_code=401, request=mock_request)

    # 2. Configure the mock to raise the properly instantiated error
    mock_openai_client.chat.completions.create.side_effect = AuthenticationError(
        message="Incorrect API key provided.", response=mock_response, body=None
    )

    # 3. Call your function and assert the outcome
    prompt_instance = OpenRouterPrompt(prompt="Prompt: {content}")
    result = prompt_instance.execute_prompt("some content")

    assert "Error: Could not get a response." in result
    assert "Incorrect API key" in result

def test_call_method_works(mocker):
    """
    Verifies that the __call__ method correctly delegates to execute_prompt.
    This is a simple test to ensure the syntactic sugar works as intended.
    """
    # Setup
    prompt_instance = OpenRouterPrompt(prompt="Test: {content}")
    
    # We can mock the instance's own method to isolate the __call__ logic
    mocker.patch.object(
        prompt_instance, 
        'execute_prompt', 
        return_value="execute_prompt was called"
    )

    # Action: Call the instance like a function
    content = "my content"
    result = prompt_instance(content)

    # Assertions
    # 1. Check that the result is passed through correctly
    assert result == "execute_prompt was called"
    
    # 2. Check that execute_prompt was called with the correct argument
    prompt_instance.execute_prompt.assert_called_once_with(content)