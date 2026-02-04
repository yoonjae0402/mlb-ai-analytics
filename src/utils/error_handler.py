
import time
import logging
from functools import wraps
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)

class VideoGenerationError(Exception):
    """Base exception for video generation errors"""
    pass

class DataFetchError(VideoGenerationError):
    """Raised when MLB API data fetch fails"""
    pass

class ValidationError(VideoGenerationError):
    """Raised when video fails quality validation"""
    pass

class AudioQualityError(VideoGenerationError):
    """Raised when audio quality is insufficient"""
    pass

class ConfigurationError(VideoGenerationError):
    """Raised when configuration is invalid"""
    pass

class ErrorHandler:
    """
    Robust error handling utilities.
    """
    
    @staticmethod
    def handle_missing_data(data_dict: Dict, required_fields: List[str], defaults: Optional[Dict] = None) -> Dict:
        """
        Handle missing data gracefully.
        Raises DataFetchError if critical fields are missing and no default provided.
        """
        if defaults is None:
            defaults = {}
            
        missing = []
        for field in required_fields:
            if field not in data_dict or data_dict[field] is None:
                if field in defaults:
                    data_dict[field] = defaults[field]
                    logger.warning(f"Missing field '{field}', using default: {defaults[field]}")
                else:
                    missing.append(field)
        
        if missing:
            raise DataFetchError(f"Missing required critical fields: {missing}")
        
        return data_dict

def retry_with_backoff(max_retries: int = 3, initial_delay: float = 1.0, backoff_factor: float = 2.0):
    """
    Decorator for retrying operations with exponential backoff.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    # Don't retry on certain fatal errors if needed (e.g. ValidationError)
                    if isinstance(e, (ValidationError, ConfigurationError)):
                        raise e
                        
                    if attempt == max_retries - 1:
                        logger.error(f"Operation {func.__name__} failed after {max_retries} attempts: {e}")
                        raise last_exception
                    
                    logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for {func.__name__}: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= backoff_factor
            return None # Should not allow
        return wrapper
    return decorator
