# tests/test_model.py

import pytest
from unittest.mock import MagicMock, patch
import httpx

# Import the specific exceptions from the openai library to simulate errors
from openai import AuthenticationError, APIConnectionError

# Import the class to be tested
# IMPORTANT: Adjust this import path to match your project structure
from llm_benchmarks.model import OpenRouterPrompt

# Helper to create a mock ChatCompletion-like object
def _create_mock_chat_completion_obj(text_content: str) -> MagicMock:
    mock_obj = MagicMock(name="MockChatCompletion")
    mock_message = MagicMock(name="MockMessage")
    mock_message.content = text_content
    mock_choice = MagicMock(name="MockChoice")
    mock_choice.message = mock_message
    mock_obj.choices = [mock_choice]
    mock_obj.id = "chatcmpl-mockid_modeltest"
    mock_obj.created = 1234567890
    mock_obj.model = "mock-model-in-test"
    mock_obj.object = "chat.completion"
    mock_obj.usage = None
    return mock_obj

# --- Test Group 1: Initialization (`__init__`) ---


def test_init_raises_error_on_bad_prompt():
    """
    Verifies that __init__ raises a ValueError if the prompt template
    is missing the required '{content}' placeholder.
    """
    with pytest.raises(
        ValueError, match="Prompt must contain a '{content}' placeholder."
    ):
        OpenRouterPrompt(prompt="This is a bad prompt without the placeholder.")


def test_init_configures_client_correctly(mocker):
    """
    Verifies that __init__ initializes the OpenAI client with the correct
    OpenRouter-specific base_url and API key.
    """
    # Mock os.getenv to return a predictable fake API key
    mocker.patch("llm_benchmarks.model.model.os.getenv", return_value="fake-api-key")

    # Mock the entire OpenAI class to capture how it's instantiated
    mock_openai_class = mocker.patch("llm_benchmarks.model.model.OpenAI")

    # Now, initialize our class
    OpenRouterPrompt(prompt="This is a valid prompt with {content}.")

    # Assert that the OpenAI client was created with the right parameters
    mock_openai_class.assert_called_once_with(
        base_url="https://openrouter.ai/api/v1", api_key="fake-api-key"
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
    # Configure the mock's create method to return our fake response object
    # The actual content can be overridden in specific tests if needed by re-mocking the return_value
    mock_client.chat.completions.create.return_value = _create_mock_chat_completion_obj(
        "This is a successful mock response."
    )

    # Patch the OpenAI class so that whenever it's called, it returns our mock_client
    mocker.patch("llm_benchmarks.model.model.OpenAI", return_value=mock_client)

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
    assert result is not None
    assert result.choices[0].message.content == "This is a successful mock response."

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
    auth_error = AuthenticationError(
        message="Incorrect API key provided.", response=mock_response, body=None
    )
    mock_openai_client.chat.completions.create.side_effect = auth_error

    # 3. Call your function and assert the outcome
    prompt_instance = OpenRouterPrompt(prompt="Prompt: {content}")

    # Expect AuthenticationError directly because it should not be retried
    with pytest.raises(AuthenticationError) as excinfo:
        prompt_instance.execute_prompt("some content")
    assert excinfo.value == auth_error


def test_call_method_works(mocker):
    """
    Verifies that the __call__ method correctly delegates to execute_prompt.
    This is a simple test to ensure the syntactic sugar works as intended.
    """
    # Setup
    # Patch openai.OpenAI to prevent actual client instantiation during OpenRouterPrompt init
    with patch("llm_benchmarks.model.model.OpenAI") as mock_openai_client_class:
        mock_openai_instance = MagicMock()
        mock_openai_client_class.return_value = mock_openai_instance

        prompt_instance = OpenRouterPrompt(prompt="Test: {content}")

        # We can mock the instance's own method to isolate the __call__ logic
        expected_text = "call content from mock object"
        mock_response_obj = _create_mock_chat_completion_obj(expected_text)

        mocker.patch.object(
            prompt_instance, 'execute_prompt', return_value=mock_response_obj
        )

        # Action: Call the instance like a function
        content_for_call = "my content for call"
        result = prompt_instance(content_for_call)

        # Assertions
        # 1. Check that execute_prompt was called with the correct argument
        prompt_instance.execute_prompt.assert_called_once_with(content_for_call)

        # 2. Check that the result is the mock object and its content is correct
        assert result is not None
        assert result.choices[0].message.content == expected_text
        # Ensure OpenAI client was attempted to be created
        mock_openai_client_class.assert_called_once()


# run tests with a real model
@pytest.mark.integration
@patch('llm_benchmarks.model.model.OpenAI') # Mock OpenAI to make this a non-live integration test
def test_openrouter_prompt_with_real_model(MockOpenAI): # Test now uses mock
    """
    An integration test to verify the OpenRouterPrompt class works with a (mocked) real model.
    OpenAI client is mocked to avoid actual network calls and ensure predictable response.
    """
    # Configure the mock client returned by OpenAI()
    mock_openai_client_instance = MockOpenAI.return_value
    expected_response_text = "Mocked response: Paris is the capital of France."
    mock_openai_client_instance.chat.completions.create.return_value = _create_mock_chat_completion_obj(
        expected_response_text
    )

    prompt_template = "Please answer the following question concisely: {content}"
    model_name = "mistralai/mistral-7b-instruct" # This model name will be passed to the mock
    prompt_instance = OpenRouterPrompt(prompt=prompt_template, model=model_name)

    question_content = "What is the capital of France?"
    response_obj = prompt_instance(question_content)

    assert response_obj is not None, "Response object should not be None."
    model_content = response_obj.choices[0].message.content

    print(
        f"\nModel Response for Mocked Integration Test (real model name): '{model_content}'"
    )

    assert model_content == expected_response_text
    assert "Paris" in model_content

    # Verify that the mock was called with the specified model name
    mock_openai_client_instance.chat.completions.create.assert_called_once()
    called_args_kwargs = mock_openai_client_instance.chat.completions.create.call_args
    assert called_args_kwargs.kwargs['model'] == model_name


@pytest.mark.integration
@patch('llm_benchmarks.model.model.OpenAI') # Mock OpenAI for this specific integration test
def test_openrouter_prompt_with_default_model(MockOpenAI): # Test now uses mock
    """
    An integration-style test to verify OpenRouterPrompt class behavior with a default model,
    but OpenAI client is mocked to avoid actual network calls and ensure predictable response.
    """
    # Configure the mock client returned by OpenAI()
    mock_openai_client_instance = MockOpenAI.return_value
    mock_openai_client_instance.chat.completions.create.return_value = _create_mock_chat_completion_obj(
        "Mocked response: The capital of France is Paris."
    )

    prompt_template = "Please answer the following question concisely: {content}"
    prompt_instance = OpenRouterPrompt(prompt=prompt_template) # Uses default model

    question_content = "What is the capital of France?"
    response_obj = prompt_instance(question_content)

    assert response_obj is not None
    response_text = response_obj.choices[0].message.content

    print(
        f"\nModel Response for Mocked Integration Test: '{response_text}'"
    )

    assert isinstance(response_text, str)
    assert "Paris" in response_text
    # Verify that the mock was called with the default model
    default_model_name = "deepseek/deepseek-r1-0528:free"
    mock_openai_client_instance.chat.completions.create.assert_called_once()
    # call_args is a tuple (args, kwargs). We need kwargs for the 'model' parameter.
    called_kwargs = mock_openai_client_instance.chat.completions.create.call_args.kwargs
    assert called_kwargs['model'] == default_model_name


# --- Test Group 3: Retry Logic ---

def test_execute_prompt_with_retry(mocker):
    """
    Tests that the execute_prompt method correctly retries on APIConnectionError
    and eventually succeeds.
    """
    # 1. Mock os.getenv to provide a dummy API key for OpenRouterPrompt initialization
    mocker.patch("llm_benchmarks.model.model.os.getenv", return_value="fake-api-key")

    # 2. Prepare mock for OpenAI client instance and its create method
    mock_openai_client_instance = MagicMock(name="MockOpenAIClientInstance")
    mock_create_method = MagicMock(name="MockCreateMethod")
    mock_openai_client_instance.chat.completions.create = mock_create_method

    # Simulate API errors for the first 2 calls, then a success
    mock_successful_response = _create_mock_chat_completion_obj("Success after retries")
    mock_create_method.side_effect = [
        APIConnectionError(request=httpx.Request(method="POST", url="http://dummy")),
        APIConnectionError(request=httpx.Request(method="POST", url="http://dummy")),
        mock_successful_response
    ]

    # 3. Patch the OpenAI class *within the module being tested* to return our instance mock
    # This ensures that when OpenRouterPrompt instantiates OpenAI, it gets our pre-configured mock
    mocker.patch("llm_benchmarks.model.model.OpenAI", return_value=mock_openai_client_instance)

    # 4. Now, setup OpenRouterPrompt instance. Its self.client will be mock_openai_client_instance
    prompt_instance = OpenRouterPrompt(prompt="Test prompt: {content}", model="test-model")

    # 5. Call the method
    content_to_pass = "some test content"
    result = prompt_instance.execute_prompt(content_to_pass)

    # 6. Assertions
    # Check that the create method was called 3 times
    assert mock_create_method.call_count == 3

    # Check that the result is the successful response
    assert result is not None
    assert result.choices[0].message.content == "Success after retries"

    # Verify the call arguments for the last successful call (optional, but good practice)
    expected_full_prompt = "Test prompt: some test content"
    mock_create_method.assert_called_with(
        model="test-model",
        messages=[{"role": "user", "content": expected_full_prompt}],
        temperature=0,
    )
