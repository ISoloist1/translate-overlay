import time
import logging
from functools import wraps


LOG_FORMAT = "%(asctime)s - [%(levelname)s] - %(message)s"
LOG_LEVEL = logging.INFO


def setup_logger():
    logger = logging.getLogger("Translate_Overlay")
    logger.setLevel(LOG_LEVEL)

    if not logger.handlers:
        formatter = logging.Formatter(LOG_FORMAT)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Optionally, create file handler
        # file_handler = logging.FileHandler("app.log")
        # file_handler.setFormatter(formatter)
        # logger.addHandler(file_handler)

    return logger


def log_timing(logger, module_name, task_name=None):
    """
    Decorator to log the execution time of a function.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            t0 = time.time()
            result = func(*args, **kwargs)
            t1 = time.time()

            func_name = task_name or func.__name__
            logger.info(f"{module_name} - {func_name}: {t1 - t0:.4f} seconds")

            return result
        
        return wrapper
    
    return decorator
