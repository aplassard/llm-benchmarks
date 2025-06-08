import logging
from tqdm import tqdm

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
        # Fallback to INFO if parsing fails, and log a warning.
        # This case should ideally be prevented by argparser choices,
        # but as a safeguard:
        root_logger.warning(f"Invalid log level string: {log_level_str}. Defaulting to INFO.")
        numeric_log_level = logging.INFO
    root_logger.setLevel(numeric_log_level)

    # Create and configure the TqdmLoggingHandler
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))

    # Clear existing handlers and add the new one
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    root_logger.addHandler(handler)

    # After setup, you might want to ensure the handler also respects the numeric_log_level
    # This is important if the handler itself had a more restrictive level set by default.
    # TqdmLoggingHandler's default level is NOTSET, so it will respect the logger's level.
    # If it were e.g. logging.WARNING, it wouldn't show INFO messages even if logger is set to INFO.
    # handler.setLevel(numeric_log_level) # Usually TqdmLoggingHandler is NOTSET by default.
                                        # If it were more restrictive, this line would be crucial.
                                        # For TqdmLoggingHandler as written, it's not strictly needed
                                        # as it defaults to NOTSET and thus inherits logger's level.
                                        # However, explicitly setting it can be a good practice for clarity
                                        # or if the handler's default might change.
                                        # For now, let's assume TqdmLoggingHandler remains NOTSET.

    # Initial log message to confirm setup (optional)
    # logging.getLogger(__name__).info(f"Logging configured with level {log_level_str.upper()}.")
