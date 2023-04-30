import os
import logging
from logging.handlers import RotatingFileHandler
import config.config as config
def configure_logging(log_dir, log_file_name, log_level=logging.INFO):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file_path = os.path.join(log_dir, log_file_name)

    # Set up the logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Set up a file handler with log rotation
    file_handler = RotatingFileHandler(
        log_file_path, maxBytes=10 * 1024 * 1024, backupCount=5
    )

    # Set up a log formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    # Optional: Add a console handler to print log messages to the console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger
