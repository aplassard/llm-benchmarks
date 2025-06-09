import pytest
import sqlite3
import os
import json
import hashlib
import threading
import time # For potential sleep in worker
from unittest.mock import patch, MagicMock

# Ensure this path allows importing from src
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from llm_benchmarks.cache.cache import CacheManager
from llm_benchmarks.solvers.gsm8k_solver import GSM8KSolver
from llm_benchmarks.model.model import OpenRouterPrompt
# Removed problematic openai.types imports


# Helper to create a mock ChatCompletion object's dictionary representation
@pytest.fixture
def cache_managers_fixture():
    DB_PATH = "test_cache.sqlite3"
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    cache_manager_enabled = CacheManager(db_path=DB_PATH, use_cache=True)
    cache_manager_enabled.init_db()
    cache_manager_disabled = CacheManager(db_path=DB_PATH, use_cache=False)

    yield cache_manager_enabled, cache_manager_disabled, DB_PATH

    cache_manager_enabled.close()
    cache_manager_disabled.close()
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

def create_mock_chat_completion_dict(content: str, usage_dict=None) -> dict:
    if usage_dict is None:
        usage_dict = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}

    return {
        "id": "chatcmpl-mockid",
        "choices": [
            { # This structure mimics ChatCompletionChoice
                "finish_reason": "stop",
                "index": 0,
                "message": { # This structure mimics ChatCompletionMessage
                    "content": content,
                    "role": "assistant",
                    "tool_calls": None,
                },
                "logprobs": None,
            }
        ],
        "created": 1677652288,
        "model": "mock-model",
        "object": "chat.completion",
        "usage": usage_dict,
        "system_fingerprint": "fp_mock"
    }

class TestCacheManager:
    def test_db_initialization(self, cache_managers_fixture):
        cache_manager_enabled, cache_manager_disabled, DB_PATH = cache_managers_fixture
        assert os.path.exists(DB_PATH)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='results';")
        assert cursor.fetchone() is not None, "Table 'results' should be created."
        conn.close()

    def test_generate_eval_id(self, cache_managers_fixture):
        cache_manager_enabled, cache_manager_disabled, DB_PATH = cache_managers_fixture
        eval_id1 = cache_manager_enabled.generate_eval_id("model1", "q1", "p1", "s1", "c1")
        eval_id2 = cache_manager_enabled.generate_eval_id("model1", "q1", "p1", "s1", "c1")
        eval_id3 = cache_manager_enabled.generate_eval_id("model2", "q1", "p1", "s1", "c1")

        assert eval_id1 == eval_id2
        assert eval_id1 != eval_id3

        expected_hash = hashlib.md5("model1-q1-p1-s1-c1".encode('utf-8')).hexdigest()
        assert eval_id1 == expected_hash

    def test_add_and_get_result_cache_enabled(self, cache_managers_fixture):
        cache_manager_enabled, cache_manager_disabled, DB_PATH = cache_managers_fixture
        eval_id = "test_eval_03"
        mock_response_dict = create_mock_chat_completion_dict("Model response content")

        cache_manager_enabled.add_result_to_cache(
            eval_id=eval_id, model_name="m1", gsm8k_question="q_content",
            prompt_template_name="p_name", gsm8k_split="test", gsm8k_config="main",
            dataset_full_expected_response="true_full", dataset_extracted_answer="true_ext",
            model_full_response_obj=mock_response_dict, # Pass the dict directly
            model_extracted_answer="model_ext", run_id="run1"
        )

        cached_result = cache_manager_enabled.get_cached_result(eval_id)
        assert cached_result is not None
        assert cached_result["model_name"] == "m1"
        assert cached_result["model_extracted_answer"] == "model_ext"

        stored_json = json.loads(cached_result["model_full_response_json"])
        assert stored_json["choices"][0]["message"]["content"] == "Model response content"

    def test_get_non_existent_result(self, cache_managers_fixture):
        cache_manager_enabled, cache_manager_disabled, DB_PATH = cache_managers_fixture
        cached_result = cache_manager_enabled.get_cached_result("non_existent_eval_id")
        assert cached_result is None

    def test_add_and_get_result_cache_disabled(self, cache_managers_fixture):
        cache_manager_enabled, cache_manager_disabled, DB_PATH = cache_managers_fixture
        eval_id = "test_eval_05"
        mock_response_dict = create_mock_chat_completion_dict("Another model response")

        cache_manager_disabled.add_result_to_cache(
            eval_id=eval_id, model_name="m2", gsm8k_question="q2_content",
            prompt_template_name="p2_name", gsm8k_split="train", gsm8k_config="socratic",
            dataset_full_expected_response="true2_full", dataset_extracted_answer="true2_ext",
            model_full_response_obj=mock_response_dict, # Pass the dict directly
            model_extracted_answer="model2_ext", run_id="run2"
        )

        retrieved_with_enabled_manager = cache_manager_enabled.get_cached_result(eval_id)
        assert retrieved_with_enabled_manager is None, "Result should not be added when cache is disabled."

        # Create a new dict for this specific test case to avoid reusing the same dict object
        another_mock_response_dict = create_mock_chat_completion_dict("Temporary model response")
        cache_manager_enabled.add_result_to_cache(
             eval_id="eval_for_disabled_get", model_name="m_temp", gsm8k_question="q_temp",
            prompt_template_name="p_temp", gsm8k_split="test", gsm8k_config="main",
            dataset_full_expected_response="true_temp", dataset_extracted_answer="true_temp_ext",
            model_full_response_obj=another_mock_response_dict, # Pass the dict directly
            model_extracted_answer="model_temp_ext", run_id="run_temp"
        )
        retrieved_with_disabled_manager = cache_manager_disabled.get_cached_result("eval_for_disabled_get")
        assert retrieved_with_disabled_manager is None, "Result should not be retrieved when cache is disabled."

    def test_06_cache_manager_thread_safety(self):
        """Tests concurrent additions and reads to the cache."""
        num_threads = 10
        threads = []
        eval_id_prefix = "thread_test_eval"

        # Worker function to be executed by each thread
        def worker(cache_manager_instance, thread_index):
            eval_id = f"{eval_id_prefix}_{thread_index}"
            model_name = f"model_thread_{thread_index}"
            question_content = f"question_thread_{thread_index}"
            prompt_name = f"prompt_thread_{thread_index}"
            split = "test"
            config = "main"
            true_full = f"true_full_thread_{thread_index}"
            true_ext = f"true_ext_thread_{thread_index}"

            # Use a unique dict for each thread's response object
            response_content = f"response_content_thread_{thread_index}"
            model_response_obj = create_mock_chat_completion_dict(response_content)

            model_ext = f"model_ext_thread_{thread_index}"
            run_id = f"run_thread_{thread_index}"

            # Add to cache
            cache_manager_instance.add_result_to_cache(
                eval_id=eval_id, model_name=model_name, gsm8k_question=question_content,
                prompt_template_name=prompt_name, gsm8k_split=split, gsm8k_config=config,
                dataset_full_expected_response=true_full, dataset_extracted_answer=true_ext,
                model_full_response_obj=model_response_obj,
                model_extracted_answer=model_ext, run_id=run_id
            )

            # Optional: short sleep to increase chance of interleaving if operations are too fast
            # time.sleep(0.01)

            # Get from cache
            cached_result = cache_manager_instance.get_cached_result(eval_id)

            self.assertIsNotNone(cached_result, f"Cache miss for eval_id {eval_id} in thread {thread_index}")
            self.assertEqual(cached_result["model_name"], model_name, f"Data mismatch for model_name in thread {thread_index}")
            self.assertEqual(cached_result["model_extracted_answer"], model_ext, f"Data mismatch for model_extracted_answer in thread {thread_index}")
            stored_json = json.loads(cached_result["model_full_response_json"])
            self.assertEqual(stored_json["choices"][0]["message"]["content"], response_content, f"Data mismatch for model_full_response_json in thread {thread_index}")

        # Create and start threads
        for i in range(num_threads):
            thread = threading.Thread(target=worker, args=(self.cache_manager_enabled, i))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Optional: Final verification - read all items and check count
        # This is a bit redundant if worker assertions are good, but can be a sanity check.
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM results WHERE eval_id LIKE '{eval_id_prefix}_%';")
        count = cursor.fetchone()[0]
        conn.close()
        self.assertEqual(count, num_threads, "Number of items in DB after threading test does not match num_threads.")


@pytest.fixture
def solver_fixture(mocker): # mocker is injected by pytest-mock
    DB_PATH = "test_solver_cache.sqlite3"
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    cache_manager = CacheManager(db_path=DB_PATH, use_cache=True)
    cache_manager.init_db()

    model_name = "test-model"
    prompt_template_content = "Question: {content} Answer:"
    prompt_template_name = "test_prompt.txt"
    data_split = "test"
    data_config = "main"
    run_id = "test_run_solver"

    mock_open_router_prompt_instance = mocker.MagicMock(spec=OpenRouterPrompt)
    mocker.patch('llm_benchmarks.solvers.gsm8k_solver.OpenRouterPrompt', return_value=mock_open_router_prompt_instance)

    solver = GSM8KSolver(
        model_name=model_name,
        prompt_template=prompt_template_content,
        cache_manager=cache_manager,
        prompt_template_name=prompt_template_name,
        data_split=data_split,
        data_config=data_config,
        run_id=run_id
    )

    yield solver, mock_open_router_prompt_instance, cache_manager, DB_PATH, model_name, prompt_template_name, data_split, data_config

    cache_manager.close()
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    # mocker.stopall() # Optional


class TestGSM8KSolverWithCache:
    def test_solve_cache_miss_then_hit(self, solver_fixture, mocker):
        solver, mock_orp_instance, cache_mgr, _, model_name, prompt_template_name, data_split, data_config = solver_fixture

        question = "What is 2 + 2?"
        true_answer_full = "The answer is #### 4"
        mock_model_response_text = "The final answer is $4"

        mock_response_payload_dict = create_mock_chat_completion_dict(mock_model_response_text)

        # Use mocker for mocks inside the test if they were previously MagicMock()
        mock_returned_chat_completion_obj = mocker.MagicMock()
        mock_returned_chat_completion_obj.model_dump.return_value = mock_response_payload_dict

        mock_message = mocker.MagicMock()
        mock_message.content = mock_model_response_text
        mock_choice = mocker.MagicMock()
        mock_choice.message = mock_message
        mock_returned_chat_completion_obj.choices = [mock_choice]

        mock_orp_instance.execute_prompt.return_value = mock_returned_chat_completion_obj

        # First call: cache miss
        result1 = solver.solve(question, true_answer_full)

        mock_orp_instance.execute_prompt.assert_called_once_with(content=question)
        assert result1.extracted_model_answer == "4"
        assert result1.model_response == mock_model_response_text

        eval_id = cache_mgr.generate_eval_id(
            model_name, question, prompt_template_name, data_split, data_config
        )
        cached_item = cache_mgr.get_cached_result(eval_id)
        assert cached_item is not None
        assert cached_item["model_extracted_answer"] == "4"

        mock_orp_instance.execute_prompt.reset_mock()

        # Second call: cache hit
        result2 = solver.solve(question, true_answer_full)

        mock_orp_instance.execute_prompt.assert_not_called()
        assert result2.extracted_model_answer == "4"
        assert result2.model_response == mock_model_response_text

    def test_solve_with_cache_disabled_in_solver(self, solver_fixture, mocker):
        solver, mock_orp_instance, cache_mgr, _, model_name, prompt_template_name, data_split, data_config = solver_fixture
        cache_mgr.use_cache = False

        question = "What is 3 + 3?"
        true_answer_full = "The answer is #### 6"
        mock_model_response_text = "The final answer is $6"

        mock_response_payload_dict = create_mock_chat_completion_dict(mock_model_response_text)

        mock_returned_chat_completion_obj = mocker.MagicMock()
        mock_returned_chat_completion_obj.model_dump.return_value = mock_response_payload_dict
        mock_message = mocker.MagicMock()
        mock_message.content = mock_model_response_text
        mock_choice = mocker.MagicMock()
        mock_choice.message = mock_message
        mock_returned_chat_completion_obj.choices = [mock_choice]

        mock_orp_instance.execute_prompt.return_value = mock_returned_chat_completion_obj

        solver.solve(question, true_answer_full)
        mock_orp_instance.execute_prompt.assert_called_once()

        solver.solve(question, true_answer_full)
        assert mock_orp_instance.execute_prompt.call_count == 2

        eval_id = cache_mgr.generate_eval_id(
            model_name, question, prompt_template_name, data_split, data_config
        )
        cache_mgr.use_cache = True # Reset for subsequent tests or direct cache checks
        cached_item = cache_mgr.get_cached_result(eval_id)
        assert cached_item is None, "Nothing should be written to cache if CacheManager.use_cache was False during solve."
