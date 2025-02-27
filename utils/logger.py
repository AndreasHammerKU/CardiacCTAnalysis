import logging

def setup_logger(debug=False):

    logger = logging.getLogger("Logger")
    
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler("log.txt")
    console_handler = logging.StreamHandler()

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    if not logger.hasHandlers():  # Avoid duplicate handlers in case of re-runs
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger