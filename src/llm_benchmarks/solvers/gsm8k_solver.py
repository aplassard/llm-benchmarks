import logging
import re
import json
from llm_benchmarks.model.model import OpenRouterPrompt
from llm_benchmarks.cache.cache import CacheManager
from openai.types.chat.chat_completion import ChatCompletion

logger = logging.getLogger(__name__)


class GSM8KResult:
    def __init__(
        self,
        question: str,
        model_response: str,
        extracted_model_answer: str | None,
        true_answer_full: str,
        extracted_true_answer: str | None,
    ):
        self.question = question
        self.model_response = model_response
        self.extracted_model_answer = extracted_model_answer
        self.true_answer_full = true_answer_full
        self.extracted_true_answer = extracted_true_answer


GSM8K_ANSWER_REGEX = r"#### (\d+\.?\d*)"

def extract_gsm8k_answer(answer_str: str) -> str | None:
    match = re.search(GSM8K_ANSWER_REGEX, answer_str)
    if match:
        return match.group(1)
    return None

class GSM8KSolver:
    def __init__(
        self,
        model_name: str,
        prompt_template: str,
        cache_manager: CacheManager | None = None, # MODIFIED
        prompt_template_name: str | None = None,   # MODIFIED
        data_split: str | None = None,             # MODIFIED
        data_config: str | None = None,            # MODIFIED
        run_id: str | None = None                  # MODIFIED
    ):
        self.model = OpenRouterPrompt(prompt=prompt_template, model=model_name)
        self.model_name = model_name
        self.cache_manager = cache_manager
        self.prompt_template_name = prompt_template_name
        self.data_split = data_split
        self.data_config = data_config
        self.run_id = run_id
        # self.prompt_template is not stored here as it's within self.model

    def _parse_cached_model_response(self, model_full_response_json_str: str | None) -> str | None:
        if not model_full_response_json_str:
            logger.warning("Cached model_full_response_json is empty or None.")
            return "Error: Empty cached response"

        try:
            full_response_obj = json.loads(model_full_response_json_str)
            if isinstance(full_response_obj, dict) and \
               full_response_obj.get("choices") and \
               isinstance(full_response_obj["choices"], list) and \
               len(full_response_obj["choices"]) > 0 and \
               isinstance(full_response_obj["choices"][0], dict) and \
               full_response_obj["choices"][0].get("message") and \
               isinstance(full_response_obj["choices"][0]["message"], dict) and \
               "content" in full_response_obj["choices"][0]["message"]:
                return full_response_obj["choices"][0]["message"]["content"]
            else:
                logger.warning(f"Cached model_full_response_json has unexpected structure. Content: {model_full_response_json_str[:200]}")
                return "Error: Invalid cached response structure"
        except json.JSONDecodeError:
            logger.warning(f"Could not parse model_full_response_json. Content: {model_full_response_json_str[:200]}")
            return "Error: Failed to parse cached response"
        except Exception as e:
            logger.error(f"Unexpected error processing cached response: {e}. Content: {model_full_response_json_str[:200]}")
            return "Error: Unexpected error processing cached response"

    def _extract_numerical_answer(self, model_response_text: str | None) -> str | None:
        if not model_response_text or model_response_text.startswith("Error:"):
            return None

        match_model = re.search(r"The final answer is \$?([\d,]*\.?\d+)", model_response_text)
        if match_model:
            extracted_answer = match_model.group(1).replace(',', '')
            if extracted_answer.endswith(".00"):
                extracted_answer = extracted_answer[:-3]
            elif extracted_answer.endswith(".0"):
                extracted_answer = extracted_answer[:-2]
            return extracted_answer
        return None

    def solve(self, question: str, true_answer_full: str) -> GSM8KResult:
        logger.debug(f"Solving question for model {self.model_name}: {question[:100]}...")
        if self.data_split and self.data_config and self.prompt_template_name: # Log only if available
            logger.debug(f"Data split: {self.data_split}, config: {self.data_config}, prompt: {self.prompt_template_name}")

        extracted_true_answer = extract_gsm8k_answer(true_answer_full)
        eval_id = None # Initialize eval_id

        # Try to use cache only if cache_manager and all necessary components for eval_id are present
        if self.cache_manager and self.prompt_template_name and self.data_split and self.data_config:
            eval_id = self.cache_manager.generate_eval_id(
                model_name=self.model_name,
                gsm8k_question_content=question,
                prompt_template_name=self.prompt_template_name,
                gsm8k_split=self.data_split,
                gsm8k_config=self.data_config,
            )
            logger.debug(f"Generated eval_id: {eval_id}")

            cached_data = self.cache_manager.get_cached_result(eval_id)
            if cached_data:
                logger.info(f"Cache hit for eval_id {eval_id}. Using cached result.")
                raw_cached_response_str = cached_data["model_full_response_json"]
                model_response_text = self._parse_cached_model_response(raw_cached_response_str)
                extracted_model_answer = cached_data["model_extracted_answer"]

                return GSM8KResult(
                    question=cached_data["gsm8k_question"],
                    model_response=model_response_text if model_response_text is not None else "Error: Processing cached response failed",
                    extracted_model_answer=extracted_model_answer,
                    true_answer_full=cached_data["dataset_full_expected_response"],
                    extracted_true_answer=cached_data["dataset_extracted_answer"],
                )

        # If self.cache_manager is None, or if any component for eval_id was None, or if it was a cache miss:
        if eval_id: # Log cache miss only if eval_id was generated (i.e., caching was attempted)
             logger.info(f"Cache miss for eval_id {eval_id}. Executing model prompt for model {self.model_name}.")
        else:
             logger.info(f"Caching not attempted or not configured. Executing model prompt for model {self.model_name}.")

        model_response_obj: ChatCompletion | None = self.model.execute_prompt(content=question)

        model_response_text: str | None = None
        if model_response_obj:
            if model_response_obj.choices and len(model_response_obj.choices) > 0:
                message = model_response_obj.choices[0].message
                if message and message.content is not None:
                    model_response_text = message.content
                else:
                    model_response_text = "Error: Message content is None in model response"
                    # Conditional logging for eval_id
                    log_prefix = f"eval_id {eval_id}: " if eval_id else ""
                    logger.warning(f"{log_prefix}Message content is None. Full response: {model_response_obj.model_dump_json(indent=2)}")
            else:
                model_response_text = "Error: No choices in model response"
                log_prefix = f"eval_id {eval_id}: " if eval_id else ""
                logger.warning(f"{log_prefix}No choices in model response. Full response: {model_response_obj.model_dump_json(indent=2)}")
        else:
            model_response_text = "Error: Model API call failed or returned None."
            log_prefix = f"eval_id {eval_id}: " if eval_id else "" # eval_id might be None here
            logger.warning(f"{log_prefix}Model API call failed (model: {self.model_name}, question: {question[:50]}...).")

        extracted_model_answer = self._extract_numerical_answer(model_response_text)

        # Add to cache only if the API call was successful, cache_manager is configured, use_cache is True,
        # and all necessary ID components are present (checked by eval_id existing) and run_id is present.
        if model_response_obj and self.cache_manager and self.cache_manager.use_cache and \
           eval_id and self.run_id : # prompt_template_name, data_split, data_config implicitly checked by eval_id
            self.cache_manager.add_result_to_cache(
                eval_id=eval_id, # This means all components for ID were present
                model_name=self.model_name,
                gsm8k_question=question,
                prompt_template_name=self.prompt_template_name,
                gsm8k_split=self.data_split,
                gsm8k_config=self.data_config,
                dataset_full_expected_response=true_answer_full,
                dataset_extracted_answer=extracted_true_answer,
                model_full_response_obj=model_response_obj.model_dump(), # Use .model_dump() for Pydantic
                model_extracted_answer=extracted_model_answer,
                run_id=self.run_id,
            )

        logger.debug(f"Question: {question}")
        logger.debug(f"Ground Truth Answer (Full): {true_answer_full}")
        logger.debug(f"Ground Truth Answer (Extracted): {extracted_true_answer}")
        logger.debug(f"Model's Full Response Text: {model_response_text}")
        logger.debug(f"Model's Predicted Answer (Extracted): {extracted_model_answer}")

        is_correct = (
            extracted_model_answer is not None
            and extracted_true_answer is not None
            and extracted_model_answer == extracted_true_answer
        )
        logger.debug(f"Result: {'CORRECT' if is_correct else 'INCORRECT'}")

        return GSM8KResult(
            question=question,
            model_response=model_response_text if model_response_text is not None else "Error: Processing failed or no response",
            extracted_model_answer=extracted_model_answer,
            true_answer_full=true_answer_full,
            extracted_true_answer=extracted_true_answer,
        )

    def __call__(self, question: str, true_answer_full: str) -> GSM8KResult:
        return self.solve(question, true_answer_full=true_answer_full)
