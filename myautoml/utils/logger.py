import logging

def setup_logger(name, log_file, level=logging.INFO):
    """Function to set up a logger."""
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    return logger

def log_message(logger, message):
    """Function to log a message."""
    logger.info(message)