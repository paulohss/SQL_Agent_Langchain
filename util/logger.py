import logging
import os
import sys
from datetime import datetime

class Logger:
    """Custom logger for SQL Agent application"""
    
    LEVELS = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    _instance = None
    
    #---------------------------------------------------------------
    # Singleton pattern to ensure only one logger instance exists
    #---------------------------------------------------------------
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    
    #---------------------------------------------------------------
    # Initialize the logger with handlers and formatters
    #---------------------------------------------------------------
    def __init__(self, level='INFO', log_to_file=True, log_file='sql_agent.log', console_output=False):
        if self._initialized:
            return
                       
        self.logger = logging.getLogger('sql_agent')
        self.logger.setLevel(self.LEVELS.get(level.upper(), logging.INFO))
        self.logger.handlers = []  # Clear existing handlers
        
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_format = logging.Formatter('%(message)s')
            console_handler.setFormatter(console_format)
            self.logger.addHandler(console_handler)
        
        
        # File handler if needed
        if log_to_file:
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
                
            file_handler = logging.FileHandler(log_file)
            file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(file_format)
            self.logger.addHandler(file_handler)
            
        self._initialized = True
    
    #---------------------------------------------------------------
    # Set logging level
    #---------------------------------------------------------------
    def setLevel(self, level):
        self.logger.setLevel(self.LEVELS.get(level.upper(), logging.INFO))
        
    #---------------------------------------------------------------
    # Log debug message
    #---------------------------------------------------------------
    def debug(self, message):
        self.logger.debug(message)
    
    #---------------------------------------------------------------
    # Log info message
    #---------------------------------------------------------------    
    def info(self, message):
        self.logger.info(message)
        
    #---------------------------------------------------------------
    # Log and Print
    #-----------------------------------------------------------------    
    def infoAndPrint(self, message):
        self.logger.info(message)        
        print(message)
    
    #---------------------------------------------------------------
    # Log warning message
    #---------------------------------------------------------------
    def warning(self, message):
        self.logger.warning(message)
    
    #---------------------------------------------------------------
    # Log error message
    #---------------------------------------------------------------
    def error(self, message):
        """Log error message"""
        self.logger.error(message)
    
    #---------------------------------------------------------------
    # Log critical message
    #---------------------------------------------------------------
    def critical(self, message):
        """Log critical message"""
        self.logger.critical(message)


# Create a default logger instance
log = Logger()