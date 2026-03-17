import logging

from app.core.config import DEBUG

def get_logger(log_level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("app_info")
    logger.setLevel(log_level)
    
    if logger.handlers:
        logger.handlers.clear()
        
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)8s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler =logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    file_handler = logging.FileHandler("model_execution.log", mode='w')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

error_logger = get_logger(logging.ERROR)
debug_logger = get_logger(logging.DEBUG)
info_logger = get_logger(logging.INFO)