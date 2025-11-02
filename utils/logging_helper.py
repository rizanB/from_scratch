import logging
import sys
from pathlib import Path

from config import Config


def setup_logger(
    name: str,
    level=None,
    log_file: str = None,
    console: bool = True,
    log_format: str = None,
    date_format: str = None,
) -> logging.Logger:
    """logger with optional file and console handler

    Args:
        name (str, optional): Logger name. Defaults to "1dcnn".
        level (int, optional): Logging level. Defaults to logging.INFO.
        log_file (str, optional): optioal path to write logs
        console (bool, optional): whether to write output to console

    Returns:
        logging.Logger: configured logger instance
    """
    if level is None:
        level = Config.LOG_LEVEL
    if log_file is None:
        log_file = Config.LOG_FILE
    if console is None:
        console = Config.LOG_TO_CONSOLE
    if log_format is None:
        log_format = Config.LOG_FORMAT
    if date_format is None:
        date_format = Config.LOG_DATE_FORMAT

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # rm duplicates
    logger.handlers.clear

    formatter = logging.Formatter(
        fmt=log_format,
        datefmt=date_format,
    )

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """get logger with default config

    Args:
        name (str): _description_

    Returns:
        logging.Logger: _description_
    """

    logger = logging.getLogger(name)

    if not logger.handlers:
        return setup_logger(name)

    return logger
