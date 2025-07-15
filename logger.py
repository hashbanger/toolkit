"""
Logger Module
"""
import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import date

from ..config.constants import constants

today_date = date.today()
current_datetime = today_date.strftime("%Y%m%d")

def setup_logging(log_filename: str=constants.Filepaths.LOGS_FILENAME):
    """Set up logging to log messages to both a file and the console, with log rotation."""
    
    log_folder = constants.Filepaths.LOGS_DIR

    log_filename = f"{log_filename}_{current_datetime}.log"

    os.makedirs(log_folder, exist_ok=True)

    log_file_path = os.path.join(log_folder, log_filename)

    # Remove any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Set up a rotating file handler
    rotating_handler = RotatingFileHandler(
        log_file_path, 
        maxBytes=100*1024*1024,  # Rotate after 100 MB
        backupCount=5            # Keep up to 5 backup log files
    )
    rotating_handler.setLevel(logging.INFO)
    rotating_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    # Set up the root logger with the rotating file handler
    logging.basicConfig(
        handlers=[rotating_handler],
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Add a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    root_logger = logging.getLogger('')
    root_logger.addHandler(console_handler)

    # Set the logging level for the 'httpx' logger to WARNING to suppress INFO logs
    httpx_logger = logging.getLogger('httpx')
    httpx_logger.setLevel(logging.WARNING)

    httpx_logger = logging.getLogger('qdrant')
    httpx_logger.setLevel(logging.WARNING)
    
    snowflake_logger = logging.getLogger('snowflake.connector')
    snowflake_logger.setLevel(logging.WARNING)

def get_logger(name):
    """Get a logger instance with the specified name"""
    return logging.getLogger(name)
