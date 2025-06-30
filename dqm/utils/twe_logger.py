"""
The confiance_logger module provides a preconfigured logger for logging messages
with specified formatting and output control. It can log messages to the standard output,
to a specified file, or both.

Usage:
    Import the module and get the default logger:
        import twe_logger
        logger = twe_logger.get_logger()

    If you need a logger with different parameters, call `get_logger` with the desired parameters:

        logger = twe_logger.get_logger(filename="my_logs.log")
        logger = twe_logger.get_logger(name="my_logger", level='debug',
        filename='my_logs.log', output="both")
    Then, use the logger within your code:

        logger.info("This is an info message")
        logger.error("This is an error message")
"""

import logging
import sys

LOGGER_DEFAULT_NAME = "twe_logger"


def log_str_to_level(str_level):
    """
    Converts a string to a corresponding logging level.

    Args:
        str_level (str): The logging level as a string.

    Returns:
        int: The corresponding logging level.
    """
    if str_level == "debug":
        level = logging.DEBUG
    elif str_level == "info":
        level = logging.INFO
    elif str_level == "warning":
        level = logging.WARNING
    elif str_level == "error":
        level = logging.ERROR
    elif str_level == "critical":
        level = logging.CRITICAL
    else:
        level = logging.NOTSET
    return level


def get_logger(name=LOGGER_DEFAULT_NAME, level="debug", filename=None, output=None):
    """
    Creates and returns a logger.

    Args:
        name (str, optional): The name of the logger.
        level (int or str, optional): The logging level.
        filename (str, optional): The name of the file where the logger should write.
        output (str, optional): Where should the logger write.
                                Can be 'stdout', 'file', or 'both'.

    Returns:
        logging.Logger: The logger.
    """
    if isinstance(level, str):
        level = log_str_to_level(level)

    # Decide the output based on whether a filename is provided and output is specified
    if output is None:
        output = "file" if filename else "stdout"

    logger = logging.getLogger(name)
    formatter = logging.Formatter(
        "[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s \
     - %(message)s",
        "%m-%d %H:%M:%S",
    )

    handlers = []

    if output in ["stdout", "both"]:
        # StreamHandler to send logs to stdout
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(level)
        handlers.append(stream_handler)

    if output in ["file", "both"] and filename:
        # FileHandler to send logs to a file
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        handlers.append(file_handler)

    if not logging.getLogger(name).hasHandlers():
        for handler in handlers:
            logger.addHandler(handler)
    else:
        logger.handlers = handlers
    logger.setLevel(level)
    logger.info("Logger: %s, handlers: %s", name, logger.handlers)
    return logger
