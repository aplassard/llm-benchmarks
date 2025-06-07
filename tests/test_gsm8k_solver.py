import pytest
from unittest.mock import MagicMock, patch
from llm_benchmarks.solvers import GSM8KSolver
from llm_benchmarks.solvers.gsm8k_solver import OpenRouterPrompt, GSM8KResult, extract_gsm8k_answer, GSM8K_ANSWER_REGEX

# Prompt template to be used in tests
PROMPT_TEMPLATE = "Solve the following math problem: {content}. The final answer is "


# --- Tests for standalone utility functions ---

@pytest.mark.parametrize(
    "answer_str, expected_extraction",
    [
        ("Question: What is 2+2? #### 4", "4"),
        ("Some text #### 123.45 and more", "123.45"),
        ("No answer here", None),
        ("#### wrong", None), # Assuming regex requires a number after ####
        ("Answer is #### 100", "100"),
        ("The answer is #### 3.0", "3.0"),
        ("The final answer is #### 123", "123"),
    ],
)
def test_extract_gsm8k_answer_logic(answer_str, expected_extraction):
    # This test uses the actual GSM8K_ANSWER_REGEX
    assert extract_gsm8k_answer(answer_str) == expected_extraction


# --- Test for GSM8KResult data class ---

def test_gsm8k_result_instantiation():
    result = GSM8KResult(
        question="q",
        model_response="mr",
        extracted_model_answer="ema",
        true_answer_full="taf",
        extracted_true_answer="eta",
    )
    assert result.question == "q"
    assert result.model_response == "mr"
    assert result.extracted_model_answer == "ema"
    assert result.true_answer_full == "taf"
    assert result.extracted_true_answer == "eta"

# --- Unit Tests with Mock Model ---

@pytest.fixture
def mock_open_router_prompt(mocker):
    mock_prompt_instance = MagicMock(spec=OpenRouterPrompt)
    mocker.patch("llm_benchmarks.solvers.gsm8k_solver.OpenRouterPrompt", return_value=mock_prompt_instance)
    return mock_prompt_instance


@pytest.mark.parametrize("verbose_flag", [True, False])
def test_gsm8k_solver_init(verbose_flag):
    # Test if GSM8KSolver initializes OpenRouterPrompt correctly and sets verbose flag
    model_name = "test/model"

    # We need to patch OpenRouterPrompt for this test, as it's instantiated in __init__
    with patch("llm_benchmarks.solvers.gsm8k_solver.OpenRouterPrompt") as mock_orp_class:
        mock_orp_instance = MagicMock()
        mock_orp_class.return_value = mock_orp_instance

        solver = GSM8KSolver(
            model_name=model_name,
            prompt_template=PROMPT_TEMPLATE,
            verbose=verbose_flag,
        )

        mock_orp_class.assert_called_once_with(
            prompt=PROMPT_TEMPLATE, model=model_name
        )
        assert solver.model is mock_orp_instance
        assert solver.verbose == verbose_flag


@pytest.mark.parametrize("verbose_flag", [True, False])
def test_solve_extracts_answer_correctly(mock_open_router_prompt, verbose_flag):
    question = "What is 2 + 2?"
    mock_response_content = "The final answer is 4"
    mock_open_router_prompt.execute_prompt.return_value = mock_response_content

    true_answer_full_dummy = "#### 0" # Extracted will be "0"

    solver = GSM8KSolver(model_name="test/model", prompt_template=PROMPT_TEMPLATE, verbose=verbose_flag)
    result = solver.solve(question, true_answer_full=true_answer_full_dummy)

    assert isinstance(result, GSM8KResult)
    assert result.question == question
    assert result.model_response == mock_response_content
    assert result.extracted_model_answer == "4"
    assert result.true_answer_full == true_answer_full_dummy
    assert result.extracted_true_answer == "0"
    mock_open_router_prompt.execute_prompt.assert_called_once_with(content=question)


@pytest.mark.parametrize("verbose_flag", [True, False])
def test_solve_extracts_answer_with_decimal_correctly(mock_open_router_prompt, verbose_flag):
    question = "What is 5 / 2?"
    mock_response_content = "The final answer is 2.5"
    mock_open_router_prompt.execute_prompt.return_value = mock_response_content

    true_answer_full_dummy = "#### 0.0"

    solver = GSM8KSolver(model_name="test/model", prompt_template=PROMPT_TEMPLATE, verbose=verbose_flag)
    result = solver.solve(question, true_answer_full=true_answer_full_dummy)

    assert isinstance(result, GSM8KResult)
    assert result.extracted_model_answer == "2.5"
    assert result.extracted_true_answer == "0.0"


# test_solve_fallback_regex_extracts_answer is removed as it's redundant.
# The current solver uses one regex for model response: r"The final answer is (\d+\.?\d*)\b"


@pytest.mark.parametrize("verbose_flag", [True, False])
def test_solve_returns_none_if_no_answer_found(mock_open_router_prompt, verbose_flag):
    question = "What is the capital of Mars?"
    mock_response_content = "There is no capital of Mars."
    mock_open_router_prompt.execute_prompt.return_value = mock_response_content

    true_answer_full_dummy = "#### N/A" # True answer might be non-numeric

    solver = GSM8KSolver(model_name="test/model", prompt_template=PROMPT_TEMPLATE, verbose=verbose_flag)
    result = solver.solve(question, true_answer_full=true_answer_full_dummy)

    assert isinstance(result, GSM8KResult)
    assert result.extracted_model_answer is None
    assert result.extracted_true_answer is None # Based on "#### N/A" and current extract_gsm8k_answer


@pytest.mark.parametrize("verbose_flag", [True, False])
def test_call_method_works(mocker, verbose_flag):
    model_name = "test/model"
    solver = GSM8KSolver(model_name=model_name, prompt_template=PROMPT_TEMPLATE, verbose=verbose_flag)

    question = "A test question"
    true_answer_full_dummy = "#### 0"

    # Prepare a mock GSM8KResult object to be returned by the mocked solve method
    mock_gsm8k_result = GSM8KResult(
        question=question,
        model_response="mock_response",
        extracted_model_answer="mock_extracted_answer",
        true_answer_full=true_answer_full_dummy,
        extracted_true_answer="0"
    )

    # Mock the solve method on the instance
    mocker.patch.object(solver, "solve", return_value=mock_gsm8k_result)

    # Use __call__
    result_from_call = solver(question, true_answer_full=true_answer_full_dummy)

    assert result_from_call is mock_gsm8k_result
    solver.solve.assert_called_once_with(question, true_answer_full=true_answer_full_dummy)


# --- Integration Test with Real Model ---


@pytest.mark.integration
@pytest.mark.parametrize("verbose_flag", [True, False])
def test_gsm8k_solver_with_real_model(verbose_flag):
    model_name = "mistralai/mistral-7b-instruct"
    prompt_for_real_model = """Question: {content}\n\nPlease provide a step-by-step solution and end your response with the phrase 'The final answer is X' where X is the numerical answer."""
    solver = GSM8KSolver(model_name=model_name, prompt_template=prompt_for_real_model, verbose=verbose_flag)

    question = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
    true_answer_full_dummy = "#### 72" # Expected: 48 + (48/2) = 72

    result = solver.solve(question, true_answer_full=true_answer_full_dummy)

    print(f"\nIntegration Test (verbose={verbose_flag}) - Question: {result.question}")
    # The solver will print details if verbose_flag is True.
    # If not, we might want to print some info here for test logs.
    if not verbose_flag:
        print(f"Integration Test - Model Response (raw): {result.model_response}")
        print(f"Integration Test - Extracted Model Answer: {result.extracted_model_answer}")
        print(f"Integration Test - Extracted True Answer: {result.extracted_true_answer}")

    assert isinstance(result, GSM8KResult)
    assert result.extracted_true_answer == "72"
    assert result.extracted_model_answer is not None, "Model answer should not be None for a solvable question."
    assert result.extracted_model_answer == "72", f"Expected '72', but got {result.extracted_model_answer}"


@pytest.mark.integration
@pytest.mark.parametrize("verbose_flag", [True, False])
def test_gsm8k_solver_real_model_complex_question(verbose_flag):
    model_name = "mistralai/mistral-7b-instruct"
    prompt_for_real_model = """Question: {content}\n\nPlease provide a step-by-step solution and end your response with the phrase 'The final answer is X' where X is the numerical answer."""
    solver = GSM8KSolver(model_name=model_name, prompt_template=prompt_for_real_model, verbose=verbose_flag)

    question = "Betty has 2 apples. She gives 1 to her friend. Then, she buys 3 more apples. How many apples does Betty have now?"
    true_answer_full_dummy = "#### 4" # Expected: 2 - 1 + 3 = 4

    result = solver.solve(question, true_answer_full=true_answer_full_dummy)

    print(f"\nIntegration Test (verbose={verbose_flag}) - Question: {result.question}")
    if not verbose_flag:
        print(f"Integration Test - Model Response (raw): {result.model_response}")
        print(f"Integration Test - Extracted Model Answer: {result.extracted_model_answer}")
        print(f"Integration Test - Extracted True Answer: {result.extracted_true_answer}")

    assert isinstance(result, GSM8KResult)
    assert result.extracted_true_answer == "4"
    assert result.extracted_model_answer is not None
    assert result.extracted_model_answer == "4", f"Expected '4', but got {result.extracted_model_answer}"
