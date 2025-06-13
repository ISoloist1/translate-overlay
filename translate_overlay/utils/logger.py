import logging


LOG_FORMAT = "%(asctime)s - [%(levelname)s] - %(module)s: %(message)s"
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

