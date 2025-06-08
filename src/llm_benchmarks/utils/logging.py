import logging
from tqdm import tqdm

class AppNameFilter(logging.Filter):
    def __init__(self, app_name=''):
        super().__init__()
        self.app_name = app_name

    def filter(self, record):
        return record.name.startswith(self.app_name)

class TqdmLoggingHandler(logging.Handler):
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

def setup_logging(log_level_str: str = "INFO"):
    """Configures the root logger with TqdmLoggingHandler."""
    root_logger = logging.getLogger()

    # Set the root logger's level
    numeric_log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    if not isinstance(numeric_log_level, int):
        # Fallback to INFO if parsing fails (e.g., invalid string from args).
        root_logger.warning(f"Invalid log level string: {log_level_str}. Defaulting to INFO.")
        numeric_log_level = logging.INFO
    root_logger.setLevel(numeric_log_level)

    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))

    handler.addFilter(AppNameFilter("llm_benchmarks"))

    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    root_logger.addHandler(handler)

    # Note: TqdmLoggingHandler defaults to level NOTSET, so it inherits the logger's level.
    # Explicitly setting handler.setLevel(numeric_log_level) is not strictly needed here
    # but can be a good practice if a handler might have a more restrictive default.
