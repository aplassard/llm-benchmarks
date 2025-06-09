import unittest
import sqlite3
import os
import json
import hashlib
from unittest.mock import patch, MagicMock

# Ensure this path allows importing from src
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from llm_benchmarks.cache.cache import CacheManager
from llm_benchmarks.solvers.gsm8k_solver import GSM8KSolver
from llm_benchmarks.model.model import OpenRouterPrompt
# Removed problematic openai.types imports


# Helper to create a mock ChatCompletion object's dictionary representation
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

class TestCacheManager(unittest.TestCase):
    DB_PATH = "test_cache.sqlite3"

    def setUp(self):
        # Ensure a clean database for each test
        if os.path.exists(self.DB_PATH):
            os.remove(self.DB_PATH)
        self.cache_manager_enabled = CacheManager(db_path=self.DB_PATH, use_cache=True)
        self.cache_manager_enabled.init_db()

        self.cache_manager_disabled = CacheManager(db_path=self.DB_PATH, use_cache=False)
        # No init_db call needed for disabled, or it should handle it gracefully

    def tearDown(self):
        self.cache_manager_enabled.close()
        self.cache_manager_disabled.close()
        if os.path.exists(self.DB_PATH):
            os.remove(self.DB_PATH)

    def test_01_db_initialization(self):
        self.assertTrue(os.path.exists(self.DB_PATH))
        conn = sqlite3.connect(self.DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='results';")
        self.assertIsNotNone(cursor.fetchone(), "Table 'results' should be created.")
        conn.close()

    def test_02_generate_eval_id(self):
        eval_id1 = self.cache_manager_enabled.generate_eval_id("model1", "q1", "p1", "s1", "c1")
        eval_id2 = self.cache_manager_enabled.generate_eval_id("model1", "q1", "p1", "s1", "c1")
        eval_id3 = self.cache_manager_enabled.generate_eval_id("model2", "q1", "p1", "s1", "c1")

        self.assertEqual(eval_id1, eval_id2)
        self.assertNotEqual(eval_id1, eval_id3)

        expected_hash = hashlib.md5("model1-q1-p1-s1-c1".encode('utf-8')).hexdigest()
        self.assertEqual(eval_id1, expected_hash)

    def test_03_add_and_get_result_cache_enabled(self):
        eval_id = "test_eval_03"
        mock_response_dict = create_mock_chat_completion_dict("Model response content")

        self.cache_manager_enabled.add_result_to_cache(
            eval_id=eval_id, model_name="m1", gsm8k_question="q_content",
            prompt_template_name="p_name", gsm8k_split="test", gsm8k_config="main",
            dataset_full_expected_response="true_full", dataset_extracted_answer="true_ext",
            model_full_response_obj=mock_response_dict, # Pass the dict directly
            model_extracted_answer="model_ext", run_id="run1"
        )

        cached_result = self.cache_manager_enabled.get_cached_result(eval_id)
        self.assertIsNotNone(cached_result)
        self.assertEqual(cached_result["model_name"], "m1")
        self.assertEqual(cached_result["model_extracted_answer"], "model_ext")

        stored_json = json.loads(cached_result["model_full_response_json"])
        self.assertEqual(stored_json["choices"][0]["message"]["content"], "Model response content")

    def test_04_get_non_existent_result(self):
        cached_result = self.cache_manager_enabled.get_cached_result("non_existent_eval_id")
        self.assertIsNone(cached_result)

    def test_05_add_and_get_result_cache_disabled(self):
        eval_id = "test_eval_05"
        mock_response_dict = create_mock_chat_completion_dict("Another model response")

        self.cache_manager_disabled.add_result_to_cache(
            eval_id=eval_id, model_name="m2", gsm8k_question="q2_content",
            prompt_template_name="p2_name", gsm8k_split="train", gsm8k_config="socratic",
            dataset_full_expected_response="true2_full", dataset_extracted_answer="true2_ext",
            model_full_response_obj=mock_response_dict, # Pass the dict directly
            model_extracted_answer="model2_ext", run_id="run2"
        )

        retrieved_with_enabled_manager = self.cache_manager_enabled.get_cached_result(eval_id)
        self.assertIsNone(retrieved_with_enabled_manager, "Result should not be added when cache is disabled.")

        # Create a new dict for this specific test case to avoid reusing the same dict object
        another_mock_response_dict = create_mock_chat_completion_dict("Temporary model response")
        self.cache_manager_enabled.add_result_to_cache(
             eval_id="eval_for_disabled_get", model_name="m_temp", gsm8k_question="q_temp",
            prompt_template_name="p_temp", gsm8k_split="test", gsm8k_config="main",
            dataset_full_expected_response="true_temp", dataset_extracted_answer="true_temp_ext",
            model_full_response_obj=another_mock_response_dict, # Pass the dict directly
            model_extracted_answer="model_temp_ext", run_id="run_temp"
        )
        retrieved_with_disabled_manager = self.cache_manager_disabled.get_cached_result("eval_for_disabled_get")
        self.assertIsNone(retrieved_with_disabled_manager, "Result should not be retrieved when cache is disabled.")


class TestGSM8KSolverWithCache(unittest.TestCase):
    DB_PATH = "test_solver_cache.sqlite3"

    def setUp(self):
        if os.path.exists(self.DB_PATH):
            os.remove(self.DB_PATH)

        self.cache_manager = CacheManager(db_path=self.DB_PATH, use_cache=True)
        self.cache_manager.init_db()

        self.model_name = "test-model"
        self.prompt_template_content = "Question: {content} Answer:"
        self.prompt_template_name = "test_prompt.txt"
        self.data_split = "test"
        self.data_config = "main"
        self.run_id = "test_run_solver"

        self.mock_open_router_prompt_instance = MagicMock(spec=OpenRouterPrompt)

        self.patcher = patch('llm_benchmarks.solvers.gsm8k_solver.OpenRouterPrompt', return_value=self.mock_open_router_prompt_instance)
        self.MockOpenRouterPrompt = self.patcher.start()

        self.solver = GSM8KSolver(
            model_name=self.model_name,
            prompt_template=self.prompt_template_content,
            cache_manager=self.cache_manager,
            prompt_template_name=self.prompt_template_name,
            data_split=self.data_split,
            data_config=self.data_config,
            run_id=self.run_id
        )

    def tearDown(self):
        self.patcher.stop()
        self.cache_manager.close()
        if os.path.exists(self.DB_PATH):
            os.remove(self.DB_PATH)

    def test_01_solve_cache_miss_then_hit(self):
        question = "What is 2 + 2?"
        true_answer_full = "The answer is #### 4"
        mock_model_response_text = "The final answer is $4"

        mock_response_payload_dict = create_mock_chat_completion_dict(mock_model_response_text)

        mock_returned_chat_completion_obj = MagicMock()
        mock_returned_chat_completion_obj.model_dump.return_value = mock_response_payload_dict

        mock_message = MagicMock()
        mock_message.content = mock_model_response_text
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_returned_chat_completion_obj.choices = [mock_choice]

        self.mock_open_router_prompt_instance.execute_prompt.return_value = mock_returned_chat_completion_obj

        # First call: cache miss
        result1 = self.solver.solve(question, true_answer_full)

        self.mock_open_router_prompt_instance.execute_prompt.assert_called_once_with(content=question)
        self.assertEqual(result1.extracted_model_answer, "4")
        self.assertEqual(result1.model_response, mock_model_response_text)

        eval_id = self.cache_manager.generate_eval_id(
            self.model_name, question, self.prompt_template_name, self.data_split, self.data_config
        )
        cached_item = self.cache_manager.get_cached_result(eval_id)
        self.assertIsNotNone(cached_item)
        self.assertEqual(cached_item["model_extracted_answer"], "4")

        self.mock_open_router_prompt_instance.execute_prompt.reset_mock()

        # Second call: cache hit
        result2 = self.solver.solve(question, true_answer_full)

        self.mock_open_router_prompt_instance.execute_prompt.assert_not_called()
        self.assertEqual(result2.extracted_model_answer, "4")
        self.assertEqual(result2.model_response, mock_model_response_text)

    def test_02_solve_with_cache_disabled_in_solver(self):
        self.cache_manager.use_cache = False

        question = "What is 3 + 3?"
        true_answer_full = "The answer is #### 6"
        mock_model_response_text = "The final answer is $6"

        mock_response_payload_dict = create_mock_chat_completion_dict(mock_model_response_text)

        mock_returned_chat_completion_obj = MagicMock()
        mock_returned_chat_completion_obj.model_dump.return_value = mock_response_payload_dict
        mock_message = MagicMock()
        mock_message.content = mock_model_response_text
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_returned_chat_completion_obj.choices = [mock_choice]

        self.mock_open_router_prompt_instance.execute_prompt.return_value = mock_returned_chat_completion_obj

        self.solver.solve(question, true_answer_full)
        self.mock_open_router_prompt_instance.execute_prompt.assert_called_once()

        self.solver.solve(question, true_answer_full)
        self.assertEqual(self.mock_open_router_prompt_instance.execute_prompt.call_count, 2)

        eval_id = self.cache_manager.generate_eval_id(
            self.model_name, question, self.prompt_template_name, self.data_split, self.data_config
        )
        self.cache_manager.use_cache = True
        cached_item = self.cache_manager.get_cached_result(eval_id)
        self.assertIsNone(cached_item, "Nothing should be written to cache if CacheManager.use_cache was False during solve.")

if __name__ == "__main__":
    unittest.main()
