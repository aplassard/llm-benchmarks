import pytest
from unittest import mock
import threading
import time
import logging # For logger instance
import uuid # For mocking run_id

# Modules to be tested or mocked
from src.llm_benchmarks.main import run_benchmarks
from src.llm_benchmarks.data import GSM8KDataset
from src.llm_benchmarks.solvers import GSM8KSolver, SolveResult # Assuming SolveResult is needed for mock
from src.llm_benchmarks.utils.args import parse_arguments # To mock its return value
from src.llm_benchmarks.utils.logging import setup_logging # To mock it
from src.llm_benchmarks.cache.cache import CacheManager # May use real or mock

# Mock data structures
class MockArgs:
    def __init__(self, num_threads, num_examples, no_cache=True,
                 log_level="INFO", prompt="default", data_split="test",
                 data_config="main", model_name="test-model",
                 prompt_template="Test template: {content}"):
        self.num_threads = num_threads
        self.num_examples = num_examples
        self.no_cache = no_cache
        self.log_level = log_level
        self.prompt = prompt
        self.data_split = data_split
        self.data_config = data_config
        self.model_name = model_name
        self.prompt_template = prompt_template # Added from args.py
        # Add any other attributes accessed in run_benchmarks or its callees if not mocked out

# Dummy example structure, adapt if GSM8KDataset returns something different
def create_dummy_example(index):
    return {"question": f"Q{index}", "answer": f"A{index} is #### {index}"}

# Dummy SolveResult, adapt to actual SolveResult structure
def create_dummy_solve_result(question, extracted_true_answer="dummy_true", model_answer="dummy_model_ans"):
    return SolveResult(
        question=question,
        true_answer_full="dummy_true_full", # Ensure all fields used by main.process_example are present
        extracted_true_answer=extracted_true_answer,
        model_response="dummy_model_response",
        extracted_model_answer=model_answer,
        error_message=None,
        prompt_template_name="dummy_template",
        start_time=0,
        end_time=0,
        time_taken=0,
        api_call_info=None
    )

@pytest.fixture
def mock_dependencies(mocker):
    """Mocks all external dependencies for run_benchmarks."""
    mocks = {}

    # Mock parse_arguments - done by caller tests directly usually by setting args
    # mocks['parse_arguments'] = mocker.patch('src.llm_benchmarks.main.parse_arguments')

    mocks['setup_logging'] = mocker.patch('src.llm_benchmarks.main.setup_logging')
    mocks['logger'] = mocker.patch('src.llm_benchmarks.main.logging.getLogger', return_value=MagicMock(spec=logging.Logger))

    # Mock GSM8KDataset
    mock_dataset_instance = MagicMock(spec=GSM8KDataset)
    # Example: make it behave like a list of 5 items
    dummy_examples = [create_dummy_example(i) for i in range(5)]
    mock_dataset_instance.__len__.return_value = len(dummy_examples)
    mock_dataset_instance.__getitem__.side_effect = lambda i: dummy_examples[i]
    mocks['GSM8KDataset'] = mocker.patch('src.llm_benchmarks.main.GSM8KDataset', return_value=mock_dataset_instance)
    mocks['dataset_instance'] = mock_dataset_instance # For direct access if needed in tests

    # Mock GSM8KSolver and its solve method
    mock_solver_instance = MagicMock(spec=GSM8KSolver)
    # Store thread IDs in a list attached to the mock_solver_instance for assertion
    mock_solver_instance.thread_ids_used = []

    def mock_solve_method(question, true_answer_full):
        # Record thread ID
        mock_solver_instance.thread_ids_used.append(threading.get_ident())
        time.sleep(0.01) # Simulate work
        # Determine if extracted_true_answer should be None based on some logic if needed for testing skips
        # For now, assume true answer is always extractable for simplicity
        extracted_true = question.split(" ")[-1] # e.g. "Q1" -> "1"
        return create_dummy_solve_result(question, extracted_true_answer=extracted_true, model_answer=extracted_true) # Simulate correct answer

    mock_solver_instance.solve.side_effect = mock_solve_method
    # Mock solver's run_id attribute if accessed by logger in run_benchmarks
    mock_solver_instance.run_id = "mock_run_id"
    mocks['GSM8KSolver'] = mocker.patch('src.llm_benchmarks.main.GSM8KSolver', return_value=mock_solver_instance)
    mocks['solver_instance'] = mock_solver_instance

    # Mock CacheManager (optional, can use real with temp for more integration-like)
    # For this unit test, focusing on concurrency, better to mock it out if it's complex
    # If run_benchmarks doesn't directly use CacheManager, but Solver does, then Solver's mock handles it.
    # The main function initializes CacheManager and passes it to solver.
    # run_benchmarks itself doesn't directly interact with cache_manager.
    # So, if solver is fully mocked, CacheManager might not need mocking here for run_benchmarks tests.

    # Mock uuid.uuid4 if run_id generation in main impacts run_benchmarks logging indirectly
    mocks['uuid4'] = mocker.patch('src.llm_benchmarks.main.uuid.uuid4', return_value=MagicMock(hex="fixed_mock_uuid"))

    return mocks

# Test for concurrent execution
def test_run_benchmarks_concurrently(mock_dependencies):
    args = MockArgs(num_threads=4, num_examples=5) # 5 examples, 4 threads

    run_benchmarks(args, mock_dependencies['logger'].return_value, mock_dependencies['dataset_instance'], mock_dependencies['solver_instance'])

    mock_solver = mock_dependencies['solver_instance']

    assert mock_solver.solve.call_count == 5 # All 5 examples should be processed

    # Check thread IDs
    thread_ids = mock_solver.thread_ids_used
    unique_thread_ids = set(thread_ids)

    # Expect multiple threads if num_examples >= num_threads > 1
    # Given 5 examples and 4 threads, we expect min(5,4) = 4 unique threads if work is distributed
    # If fewer examples than threads, then number of unique threads = number of examples
    expected_unique_threads = min(args.num_examples, args.num_threads)
    if args.num_examples > 0 and args.num_threads > 1:
         assert len(unique_thread_ids) >= 1 # Should be > 1, but depends on timing and Python's GIL.
                                           # With sleep, it's very likely to be > 1.
                                           # A more robust check might be that not all IDs are the same if len > 1
         assert len(unique_thread_ids) <= expected_unique_threads
         if len(thread_ids) > 1 and len(unique_thread_ids) == 1:
             print(f"Warning: Only one thread ID ({unique_thread_ids.pop()}) was used despite num_threads={args.num_threads} and num_examples={args.num_examples}. This might indicate an issue or a very fast execution not yielding to other threads.")
         # For this specific case (5 examples, 4 threads), we expect up to 4 threads.
         # It's possible that with very fast tasks, not all threads in the pool get used.
         # A more reliable check is that more than one was used if conditions apply.
         if args.num_examples >= args.num_threads and args.num_threads > 1:
             assert len(unique_thread_ids) > 1, "Expected multiple threads to be used."
         elif args.num_examples < args.num_threads and args.num_examples > 1:
             assert len(unique_thread_ids) > 1, "Expected multiple threads to be used when examples > 1 and < num_threads."

    elif args.num_examples > 0 and args.num_threads == 1:
        assert len(unique_thread_ids) == 1 # Only one thread should be used


# Test for single thread execution
def test_run_benchmarks_single_thread(mock_dependencies):
    args = MockArgs(num_threads=1, num_examples=3) # 3 examples, 1 thread

    run_benchmarks(args, mock_dependencies['logger'].return_value, mock_dependencies['dataset_instance'], mock_dependencies['solver_instance'])

    mock_solver = mock_dependencies['solver_instance']

    assert mock_solver.solve.call_count == 3

    thread_ids = mock_solver.thread_ids_used
    unique_thread_ids = set(thread_ids)

    if args.num_examples > 0: # Only assert if work was done
        assert len(unique_thread_ids) == 1 # Only one thread ID expected


# Test for num_examples = 0
def test_run_benchmarks_zero_examples(mock_dependencies):
    args = MockArgs(num_threads=4, num_examples=0)

    run_benchmarks(args, mock_dependencies['logger'].return_value, mock_dependencies['dataset_instance'], mock_dependencies['solver_instance'])

    mock_solver = mock_dependencies['solver_instance']
    assert mock_solver.solve.call_count == 0
    # Logger info should be called about no examples
    mock_dependencies['logger'].return_value.info.assert_any_call("No examples to run. The dataset might be empty or num_examples is 0.")

# Test for num_examples = -1 (all examples)
def test_run_benchmarks_all_examples(mock_dependencies):
    num_dataset_examples = len(mock_dependencies['dataset_instance'])
    args = MockArgs(num_threads=2, num_examples=-1) # All examples, 2 threads

    run_benchmarks(args, mock_dependencies['logger'].return_value, mock_dependencies['dataset_instance'], mock_dependencies['solver_instance'])

    mock_solver = mock_dependencies['solver_instance']
    assert mock_solver.solve.call_count == num_dataset_examples

    thread_ids = mock_solver.thread_ids_used
    unique_thread_ids = set(thread_ids)

    if num_dataset_examples > 0 and args.num_threads > 1:
        expected_threads = min(num_dataset_examples, args.num_threads)
        if num_dataset_examples >= expected_threads and expected_threads > 1:
             assert len(unique_thread_ids) > 1, "Expected multiple threads for multiple examples and threads > 1"
        elif num_dataset_examples > 1 and expected_threads == 1: # e.g. 5 examples, 1 thread
             assert len(unique_thread_ids) == 1
    elif num_dataset_examples > 0 and args.num_threads == 1:
        assert len(unique_thread_ids) == 1

# Test when all examples are skipped due to GT extraction failure
@mock.patch('src.llm_benchmarks.main.process_example') # Mock the helper directly
def test_run_benchmarks_all_skipped_gt_extraction(mock_process_example, mock_dependencies):
    args = MockArgs(num_threads=2, num_examples=3)

    # Configure process_example mock to simulate all examples being skipped
    # process_example returns: (is_correct, was_skipped, solve_result_or_none)
    mock_process_example.return_value = (False, True, None)

    run_benchmarks(args, mock_dependencies['logger'].return_value, mock_dependencies['dataset_instance'], mock_dependencies['solver_instance'])

    # Assert that process_example was called for each example
    assert mock_process_example.call_count == args.num_examples

    # Assert that solver.solve was NOT called directly from run_benchmarks loop,
    # as process_example is now the unit of work submitted to executor.
    # Solver is passed to process_example, so its 'solve' would be called *inside* process_example.
    # Here, since we mocked process_example itself, solver.solve won't be called from the perspective of this test.
    mock_dependencies['solver_instance'].solve.assert_not_called()

    # Check for appropriate logging
    # Total examples attempted: 3
    # Examples skipped (GT extraction failed): 3
    # Total examples with valid ground truth: 0
    # Accuracy: (not logged or 0 based on 0 valid)
    # "No examples were processed that had extractable ground truth answers."

    # Construct expected log messages based on the summary in main.py
    logger_mock = mock_dependencies['logger'].return_value

    # Check that specific info lines were logged
    # This needs to be specific to what your logger actually outputs.
    # Using assert_any_call is safer if order or other logs are present.
    logger_mock.info.assert_any_call("\n--- Benchmark Summary ---")
    logger_mock.info.assert_any_call(f"Total examples attempted: {args.num_examples}")
    logger_mock.info.assert_any_call(f"Examples skipped (GT extraction failed): {args.num_examples}")
    logger_mock.info.assert_any_call(f"Total examples with valid ground truth: 0")
    logger_mock.info.assert_any_call("\nNo examples were processed that had extractable ground truth answers.")

    # Ensure accuracy isn't logged or logged as 0 if total_processed_valid_gt is 0
    # Check that a line starting with "Accuracy" was NOT called or was called with 0/NaN.
    # Current code path for 0 valid GT examples logs the "No examples were processed..." message instead of accuracy.
    accuracy_log_found = False
    for call_args in logger_mock.info.call_args_list:
        if "Accuracy" in call_args[0][0]:
            accuracy_log_found = True
            break
    assert not accuracy_log_found, "Accuracy should not be logged when all examples are skipped."

# (Optional) Add a test for when num_examples > len(dataset)
# (Optional) Add a test for when a specific error occurs during future.result() in run_benchmarks

# Note: The MagicMock for the logger will capture calls.
# You can assert specific log messages if needed, e.g.,
# mock_logger_instance = mock_dependencies['logger']
# mock_logger_instance.info.assert_any_call("Running benchmark on 5 examples using 4 threads...")
# mock_logger_instance.info.assert_any_call("\n--- Benchmark Summary ---")
# ... etc.
# Consider also testing the content of what's logged for correct/skipped counts.
# For example, if one was correct and one skipped:
# logger_mock.info.assert_any_call(f"Total examples with valid ground truth: {num_examples - num_skipped}")
# logger_mock.info.assert_any_call(f"Correct answers: {num_correct}")

# This structure relies on the mock_dependencies fixture to set up all common mocks.
# Individual tests then specify the MockArgs and assert outcomes.
# The GSM8KSolver's 'solve' method is the key part that records thread IDs.
# The GSM8KDataset mock provides the data.
# Logger and setup_logging are mocked to avoid side effects.
# uuid.uuid4 is mocked for deterministic run_ids if they affect logging/output.
# CacheManager is not directly used by run_benchmarks, so it might not need explicit mocking
# here if the solver (which uses it) is already fully mocked.
# If the real solver were used, then CacheManager would need mocking or a temp DB.
# Since solver is mocked, its interaction with CacheManager is bypassed.
# Prompt loading is part of args parsing, which we are controlling via MockArgs.
# The prompt_template attribute in MockArgs should be set.

# To make thread ID assertion more robust:
# In test_run_benchmarks_concurrently:
# if args.num_examples > 1 and args.num_threads > 1:
#     if args.num_examples >= args.num_threads:
#         assert len(unique_thread_ids) > 1, "Expected multiple threads for num_examples >= num_threads > 1"
#     else: # num_examples < num_threads
#         assert len(unique_thread_ids) == args.num_examples, "Expected one thread per example when num_examples < num_threads"
# elif args.num_examples == 1 or args.num_threads == 1:
#     assert len(unique_thread_ids) == 1, "Expected single thread for 1 example or 1 thread"
# else: # num_examples == 0
#     assert len(unique_thread_ids) == 0, "Expected no threads for 0 examples"

# The logic for expected unique threads in the test_run_benchmarks_concurrently can be refined.
# If num_threads=4, num_examples=5: expect up to 4 unique threads.
# If num_threads=4, num_examples=2: expect 2 unique threads.
# If num_threads=1, num_examples=5: expect 1 unique thread.
# The current assertions for thread_ids are a bit loose, this can be tightened.
# For example, if num_examples=5, num_threads=4, we'd expect min(5,4) = 4 unique threads if tasks are well distributed.
# The test 'test_run_benchmarks_concurrently' has been updated with a more specific check.
# The test 'test_run_benchmarks_all_examples' also has refined thread checking.

# The test 'test_run_benchmarks_all_skipped_gt_extraction' mocks 'process_example' directly
# which is a good way to test how 'run_benchmarks' aggregates results from the futures.
# This means 'solver.solve' is not called in this specific test, which is fine as we are testing
# the aggregation logic of run_benchmarks itself.

# Final check on MockArgs, ensure all necessary fields from 'args' that are used in 'run_benchmarks'
# and 'process_example' (if not mocked) are present.
# 'args.prompt_template' is used by solver, but solver is mocked.
# 'args.log_level' used by setup_logging (mocked).
# 'args.model_name' used by solver (mocked).
# 'args.num_threads' used by run_benchmarks.
# 'args.num_examples' used by run_benchmarks.
# 'args.data_split', 'args.data_config' used by solver (mocked).
# 'args.no_cache' used by main to set up solver (solver mocked).
# Seems okay for now.
