import logging
import os
import sys
from typing import Optional
from config.config_utils import get_config

def setup_logger(name: str, log_file: Optional[str] = None, level: Optional[str] = None) -> logging.Logger:
    """
    Set up and return a logger instance.
    
    Args:
        name: Name of the logger
        log_file: Path to log file
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        A configured logger instance
    """
    config = get_config()
    
    if level is None:
        level = config.get("logging.level", "INFO")
    
    if log_file is None:
        log_file = config.get("logging.file")
        
    # Create logs directory if it doesn't exist
    if log_file and not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))
    
    # Set up logger
    logger = logging.getLogger(name)
    
    # Convert string level to logging level constant
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    
    logger.setLevel(numeric_level)
    
    # Remove existing handlers if any
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        config.get("logging.format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    
    # Add file handler if log file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

# Create default application logger
app_logger = setup_logger("teacher_tester")