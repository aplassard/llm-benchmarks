import sqlite3
import json
import logging
import hashlib
from datetime import datetime

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self, db_path: str, use_cache: bool = True):
        self.db_path = db_path
        self.use_cache = use_cache
        self._conn = None
        if self.use_cache:
            self._connect()

    def _connect(self):
        try:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.row_factory = sqlite3.Row # Allows accessing columns by name
        except sqlite3.Error as e:
            logger.error(f"Error connecting to SQLite database {self.db_path}: {e}")
            self._conn = None # Ensure connection is None if it fails
            self.use_cache = False # Disable cache if connection fails
            logger.warning(f"Caching disabled due to database connection error.")


    def init_db(self):
        if not self.use_cache or not self._conn:
            logger.info("Cache is disabled or connection not established. Skipping DB initialization.")
            return

        try:
            with self._conn:
                self._conn.execute("""
                CREATE TABLE IF NOT EXISTS results (
                    eval_id TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    gsm8k_question TEXT NOT NULL,
                    prompt_template_name TEXT NOT NULL,
                    gsm8k_split TEXT NOT NULL,
                    gsm8k_config TEXT NOT NULL,
                    dataset_full_expected_response TEXT NOT NULL,
                    dataset_extracted_answer TEXT,
                    model_full_response_json TEXT,
                    model_extracted_answer TEXT,
                    run_id TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """)
                logger.info(f"Database initialized and 'results' table ensured at {self.db_path}")
        except sqlite3.Error as e:
            logger.error(f"Error initializing database table: {e}")
            self._conn = None # Ensure connection is None if it fails
            self.use_cache = False # Disable cache if table creation fails
            logger.warning(f"Caching disabled due to database initialization error.")

    def generate_eval_id(self, model_name: str, gsm8k_question_content: str, prompt_template_name: str, gsm8k_split: str, gsm8k_config: str) -> str:
        # Create a stable string representation of the inputs
        unique_string = f"{model_name}-{gsm8k_question_content}-{prompt_template_name}-{gsm8k_split}-{gsm8k_config}"
        return hashlib.md5(unique_string.encode('utf-8')).hexdigest()

    def get_cached_result(self, eval_id: str) -> dict | None:
        if not self.use_cache or not self._conn:
            return None

        try:
            with self._conn:
                cursor = self._conn.execute("SELECT * FROM results WHERE eval_id = ?", (eval_id,))
                row = cursor.fetchone()
                if row:
                    logger.info(f"Cache hit for eval_id: {eval_id}")
                    return dict(row)
                else:
                    logger.info(f"Cache miss for eval_id: {eval_id}")
                    return None
        except sqlite3.Error as e:
            logger.error(f"Error fetching from cache for eval_id {eval_id}: {e}")
            return None

    def add_result_to_cache(
        self,
        eval_id: str,
        model_name: str,
        gsm8k_question: str, # This is the specific question text from the dataset
        prompt_template_name: str,
        gsm8k_split: str,
        gsm8k_config: str,
        dataset_full_expected_response: str,
        dataset_extracted_answer: str | None,
        model_full_response_obj: dict, # Assuming this is the raw response object from OpenAI
        model_extracted_answer: str | None,
        run_id: str,
    ):
        if not self.use_cache or not self._conn:
            return

        model_full_response_json = json.dumps(model_full_response_obj)

        try:
            with self._conn:
                self._conn.execute("""
                INSERT INTO results (
                    eval_id, model_name, gsm8k_question, prompt_template_name, gsm8k_split, gsm8k_config,
                    dataset_full_expected_response, dataset_extracted_answer,
                    model_full_response_json, model_extracted_answer, run_id, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    eval_id, model_name, gsm8k_question, prompt_template_name, gsm8k_split, gsm8k_config,
                    dataset_full_expected_response, dataset_extracted_answer,
                    model_full_response_json, model_extracted_answer, run_id, datetime.now()
                ))
                logger.info(f"Result added to cache with eval_id: {eval_id}")
        except sqlite3.Error as e:
            logger.error(f"Error adding to cache for eval_id {eval_id}: {e}")

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.info("Cache database connection closed.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
