# tests/test_model.py

import pytest
from unittest.mock import MagicMock, patch
import httpx

# Import the specific exceptions from the openai library to simulate errors
from openai import AuthenticationError

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
    mock_openai_client.chat.completions.create.side_effect = AuthenticationError(
        message="Incorrect API key provided.", response=mock_response, body=None
    )

    # 3. Call your function and assert the outcome
    prompt_instance = OpenRouterPrompt(prompt="Prompt: {content}")
    result = prompt_instance.execute_prompt("some content")

    assert result is None


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


# --- Test Group 3: Retries and Error Handling (`execute_prompt` with backoff) ---
from openai import APIStatusError, APITimeoutError, APIConnectionError, RateLimitError
import time # For time.sleep
import random # For random.uniform

# Need to import logger from the module to mock it, if it's used there.
# from llm_benchmarks.model import model as model_module_logger # if logger is model_module.logger
# Assuming logger is obtained by `logging.getLogger(__name__)` in model.py,
# patching 'llm_benchmarks.model.model.logger' is correct.

class TestOpenRouterPromptWithRetries:
    PROMPT_TEMPLATE = "Test prompt: {content}"
    MODEL_NAME = "test-model/retry-model"
    DEFAULT_CONTENT = "some content"

    @pytest.fixture(autouse=True)
    def setup_mocks(self, mocker):
        # Mock os.getenv for API key
        mocker.patch("llm_benchmarks.model.model.os.getenv", return_value="fake-api-key-for-retry-tests")

        # Mock OpenAI client parts
        self.mock_openai_client_instance = MagicMock()
        self.mock_chat_completions_create = self.mock_openai_client_instance.chat.completions.create
        mocker.patch("llm_benchmarks.model.model.OpenAI", return_value=self.mock_openai_client_instance)

        # Mock time.sleep
        self.mock_sleep = mocker.patch("llm_benchmarks.model.model.time.sleep")

        # Mock random.uniform for predictable jitter (return 0 for no jitter in tests)
        self.mock_random_uniform = mocker.patch("llm_benchmarks.model.model.random.uniform", return_value=0.0)

        # Mock the logger used in model.py
        self.mock_logger = mocker.patch("llm_benchmarks.model.model.logger")

        # Instantiate the class under test
        self.prompt_instance = OpenRouterPrompt(prompt=self.PROMPT_TEMPLATE, model=self.MODEL_NAME)

        # Ensure retry parameters are at known values for tests
        self.prompt_instance.MAX_RETRIES = 3 # Override for faster tests
        self.prompt_instance.INITIAL_DELAY = 0.1 # s, for faster tests
        self.prompt_instance.BACKOFF_FACTOR = 2.0
        self.prompt_instance.JITTER_FACTOR = 0.1 # Though random.uniform is mocked to 0

    def _get_mock_response(self, text_content="successful response"):
        return _create_mock_chat_completion_obj(text_content)

    def test_execute_prompt_success_first_try(self):
        expected_response = self._get_mock_response()
        self.mock_chat_completions_create.return_value = expected_response

        result = self.prompt_instance.execute_prompt(self.DEFAULT_CONTENT)

        assert result == expected_response
        self.mock_chat_completions_create.assert_called_once()
        self.mock_sleep.assert_not_called()
        self.mock_logger.warning.assert_not_called()
        self.mock_logger.error.assert_not_called()

    def test_execute_prompt_retry_once_then_succeed_ratelimit(self):
        expected_response = self._get_mock_response()
        mock_request = httpx.Request(method="POST", url="https://api.test.com") # Needed for RateLimitError
        mock_http_response = httpx.Response(status_code=429, request=mock_request) # Needed for RateLimitError

        self.mock_chat_completions_create.side_effect = [
            RateLimitError(message="Too many requests", response=mock_http_response, body=None),
            expected_response
        ]

        result = self.prompt_instance.execute_prompt(self.DEFAULT_CONTENT)

        assert result == expected_response
        assert self.mock_chat_completions_create.call_count == 2
        self.mock_sleep.assert_called_once_with(self.prompt_instance.INITIAL_DELAY) # Jitter is mocked to 0
        self.mock_logger.warning.assert_called_once()
        assert "Retryable API error" in self.mock_logger.warning.call_args[0][0]
        assert "RateLimitError" in self.mock_logger.warning.call_args[0][0]

    def test_execute_prompt_retry_max_attempts_then_fail_apitimeout(self):
        mock_request = httpx.Request(method="POST", url="https://api.test.com")
        errors = [APITimeoutError(message="Timeout", request=mock_request) for _ in range(self.prompt_instance.MAX_RETRIES)]
        self.mock_chat_completions_create.side_effect = errors

        result = self.prompt_instance.execute_prompt(self.DEFAULT_CONTENT)

        assert result is None
        assert self.mock_chat_completions_create.call_count == self.prompt_instance.MAX_RETRIES
        assert self.mock_sleep.call_count == self.prompt_instance.MAX_RETRIES - 1

        # Check exponential backoff (simplified check, assumes no jitter due to mocking)
        expected_delays = [self.prompt_instance.INITIAL_DELAY * (self.prompt_instance.BACKOFF_FACTOR ** i) for i in range(self.prompt_instance.MAX_RETRIES - 1)]
        for i, call_args in enumerate(self.mock_sleep.call_args_list):
            assert call_args[0][0] == expected_delays[i]

        self.mock_logger.error.assert_called()
        # The last log before final error is a warning for the retry attempt, then a final error
        assert f"API call failed for model {self.MODEL_NAME} after {self.prompt_instance.MAX_RETRIES} retries." in self.mock_logger.error.call_args_list[-1][0][0]


    def test_execute_prompt_non_retryable_api_status_error_400(self):
        mock_request = httpx.Request(method="POST", url="https.api.test.com")
        mock_response = httpx.Response(status_code=400, request=mock_request, json={"error": "bad request"})
        self.mock_chat_completions_create.side_effect = APIStatusError(message="Bad request", response=mock_response, body=None)

        result = self.prompt_instance.execute_prompt(self.DEFAULT_CONTENT)

        assert result is None
        self.mock_chat_completions_create.assert_called_once()
        self.mock_sleep.assert_not_called()
        self.mock_logger.error.assert_called_once()
        assert "Non-retryable APIStatusError" in self.mock_logger.error.call_args[0][0]
        assert "Status 400" in self.mock_logger.error.call_args[0][0]

    def test_execute_prompt_retryable_api_status_error_500_then_succeed(self):
        expected_response = self._get_mock_response()
        mock_request = httpx.Request(method="POST", url="https.api.test.com")
        mock_response_500 = httpx.Response(status_code=500, request=mock_request, json={"error": "server error"})

        self.mock_chat_completions_create.side_effect = [
            APIStatusError(message="Server error", response=mock_response_500, body=None),
            expected_response
        ]

        result = self.prompt_instance.execute_prompt(self.DEFAULT_CONTENT)

        assert result == expected_response
        assert self.mock_chat_completions_create.call_count == 2
        self.mock_sleep.assert_called_once_with(self.prompt_instance.INITIAL_DELAY) # Jitter mocked to 0
        self.mock_logger.warning.assert_called_once()
        assert "Retryable API StatusError" in self.mock_logger.warning.call_args[0][0]
        assert "Status 500" in self.mock_logger.warning.call_args[0][0]

    def test_execute_prompt_unexpected_exception_then_fail(self):
        # Test that a generic Exception is caught and retried (as per current implementation)
        # and eventually fails after max retries.
        errors = [Exception("Some very unexpected error") for _ in range(self.prompt_instance.MAX_RETRIES)]
        self.mock_chat_completions_create.side_effect = errors

        result = self.prompt_instance.execute_prompt(self.DEFAULT_CONTENT)
        assert result is None
        assert self.mock_chat_completions_create.call_count == self.prompt_instance.MAX_RETRIES
        assert self.mock_sleep.call_count == self.prompt_instance.MAX_RETRIES -1

        # Check that the first attempt logs the unexpected error
        first_log_call_args = self.mock_logger.error.call_args_list[0][0][0]
        assert f"An unexpected error occurred with model {self.MODEL_NAME}" in first_log_call_args
        assert "Exception - Some very unexpected error" in first_log_call_args

        # Check that the final error log indicates failure after retries
        final_log_call_args = self.mock_logger.error.call_args_list[-1][0][0]
        assert f"API call failed for model {self.MODEL_NAME} after {self.prompt_instance.MAX_RETRIES} retries" in final_log_call_args

        # Also check that a warning about retrying an unexpected error is logged for each retry attempt (if applicable by design)
        # The current code logs a generic "Retrying API error" or specific status error for warnings.
        # For unexpected errors, it logs an error then proceeds to retry.
        # Let's verify the warnings for retrying. The warnings are for retryable errors.
        # The logger.info message "Waiting ...s before next retry" should be called.
        assert self.mock_logger.info.call_count == self.prompt_instance.MAX_RETRIES -1
        assert "Waiting" in self.mock_logger.info.call_args_list[0][0][0]
