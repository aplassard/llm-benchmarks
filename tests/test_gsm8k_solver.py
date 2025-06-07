import pytest
from unittest.mock import MagicMock
from llm_benchmarks.gsm8k_solver import GSM8KSolver
from llm_benchmarks.model.model import OpenRouterPrompt

# Prompt template to be used in tests
PROMPT_TEMPLATE = "Solve the following math problem: {content}. The final answer is "

# --- Unit Tests with Mock Model ---


@pytest.fixture
def mock_open_router_prompt(mocker):
    # Create a mock for the OpenRouterPrompt class
    mock_prompt_instance = MagicMock(spec=OpenRouterPrompt)

    # Patch OpenRouterPrompt in the gsm8k_solver module where it's used
    mocker.patch(
        "llm_benchmarks.gsm8k_solver.OpenRouterPrompt",
        return_value=mock_prompt_instance,
    )
    return mock_prompt_instance


def test_gsm8k_solver_init(mock_open_router_prompt):
    # Test if GSM8KSolver initializes OpenRouterPrompt correctly
    model_name = "test/model"
    solver = GSM8KSolver(model_name=model_name, prompt_template=PROMPT_TEMPLATE)

    # Check if OpenRouterPrompt was called with the correct arguments
    # mock_open_router_prompt_class = solver.model # This is the instance - currently unused

    # We need to assert that the class was called correctly to create the instance
    # The mock_open_router_prompt fixture itself is the instance,
    # so we check its constructor mock if available, or assert on its properties
    # For this setup, we check that the constructor of OpenRouterPrompt was called with right args
    # This requires OpenRouterPrompt itself to be patched, which is what the fixture does.

    # Get the class constructor mock that was patched by the fixture
    # The path for patching is 'llm_benchmarks.gsm8k_solver.OpenRouterPrompt'
    # This is where GSM8KSolver looks up OpenRouterPrompt

    # Assert that OpenRouterPrompt was called once with these arguments
    # The mock_open_router_prompt fixture represents the *instance* returned by the patched class.
    # To check the class constructor call, we need to look at the patched class itself.
    # Let's adjust the fixture slightly to make this easier or assert differently.

    # The fixture `mock_open_router_prompt` is the *instance*.
    # The class `llm_benchmarks.gsm8k_solver.OpenRouterPrompt` was patched.
    # So, we can assert that the patched class was called correctly.
    from llm_benchmarks.gsm8k_solver import OpenRouterPrompt as PatchedOpenRouterPrompt

    PatchedOpenRouterPrompt.assert_called_once_with(
        prompt=PROMPT_TEMPLATE, model=model_name
    )

    # Additionally, assert that the model instance on the solver is the one returned by the mock
    assert solver.model is mock_open_router_prompt


def test_solve_extracts_answer_correctly(mock_open_router_prompt):
    # Test if the solve method extracts the answer correctly using the primary regex
    question = "What is 2 + 2?"
    mock_response_content = "The final answer is 4"
    mock_open_router_prompt.execute_prompt.return_value = mock_response_content

    solver = GSM8KSolver(model_name="test/model", prompt_template=PROMPT_TEMPLATE)
    answer = solver.solve(question)

    assert answer == "4"
    mock_open_router_prompt.execute_prompt.assert_called_once_with(content=question)


def test_solve_extracts_answer_with_decimal_correctly(mock_open_router_prompt):
    # Test if the solve method extracts decimal answers correctly
    question = "What is 5 / 2?"
    mock_response_content = "The final answer is 2.5"
    mock_open_router_prompt.execute_prompt.return_value = mock_response_content

    solver = GSM8KSolver(model_name="test/model", prompt_template=PROMPT_TEMPLATE)
    answer = solver.solve(question)

    assert answer == "2.5"


def test_solve_fallback_regex_extracts_answer(mock_open_router_prompt):
    # Test if the fallback regex extracts an answer if the primary one fails
    question = "What is 3 * 3?"
    # Response that doesn't match "The final answer is X"
    mock_response_content = "The result is 9."
    mock_open_router_prompt.execute_prompt.return_value = mock_response_content

    solver = GSM8KSolver(model_name="test/model", prompt_template=PROMPT_TEMPLATE)
    answer = solver.solve(question)

    assert answer == "9"  # Fallback should find the number 9


def test_solve_returns_none_if_no_answer_found(mock_open_router_prompt):
    # Test if solve returns None when no number is found in the response
    question = "What is the capital of Mars?"
    mock_response_content = "There is no capital of Mars."
    mock_open_router_prompt.execute_prompt.return_value = mock_response_content

    solver = GSM8KSolver(model_name="test/model", prompt_template=PROMPT_TEMPLATE)
    answer = solver.solve(question)

    assert answer is None


def test_call_method_works(mocker):
    # Test if the __call__ method delegates to solve
    solver = GSM8KSolver(model_name="test/model", prompt_template=PROMPT_TEMPLATE)

    # Mock the solve method on the instance
    mocker.patch.object(solver, "solve", return_value="mocked_answer")

    question = "A test question"
    result = solver(question)  # Use __call__

    assert result == "mocked_answer"
    solver.solve.assert_called_once_with(question)


# --- Integration Test with Real Model ---


@pytest.mark.integration
def test_gsm8k_solver_with_real_model():
    # Integration test with a real model (requires API key and network)
    # Ensure OPENROUTER_API_KEY is set in the environment for this test

    # Using a known free and fast model for testing
    # You can change this to any model available on OpenRouter
    model_name = "mistralai/mistral-7b-instruct"

    # A prompt that guides the model to output in the desired format
    # This template is crucial for the regex to work reliably.
    prompt_for_real_model = """Question: {content}

Please provide a step-by-step solution and end your response with the phrase 'The final answer is X' where X is the numerical answer."""

    solver = GSM8KSolver(model_name=model_name, prompt_template=prompt_for_real_model)

    # A simple GSM8K-style question
    question = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    # Expected answer: 48 + (48/2) = 48 + 24 = 72

    answer = solver.solve(question)

    print(f"\nIntegration Test - Question: {question}")
    print(
        f"Integration Test - Model Response (raw, for debugging): {solver.model.execute_prompt(content=question)}"
    )
    print(f"Integration Test - Extracted Answer: {answer}")

    assert answer is not None, "Answer should not be None for a solvable question."
    # We expect the answer to be a string, so compare with "72"
    assert answer == "72", f"Expected '72', but got {answer}"


@pytest.mark.integration
def test_gsm8k_solver_real_model_complex_question():
    model_name = "mistralai/mistral-7b-instruct"
    prompt_for_real_model = """Question: {content}

Please provide a step-by-step solution and end your response with the phrase 'The final answer is X' where X is the numerical answer."""
    solver = GSM8KSolver(model_name=model_name, prompt_template=prompt_for_real_model)

    # A slightly more complex question
    question = "Betty has 2 apples. She gives 1 to her friend. Then, she buys 3 more apples. How many apples does Betty have now?"
    # Expected: 2 - 1 + 3 = 4

    answer = solver.solve(question)

    print(f"\nIntegration Test - Question: {question}")
    print(
        f"Integration Test - Model Response (raw, for debugging): {solver.model.execute_prompt(content=question)}"
    )
    print(f"Integration Test - Extracted Answer: {answer}")

    assert answer is not None
    assert answer == "4", f"Expected '4', but got {answer}"
