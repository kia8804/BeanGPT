"""
Custom exceptions for the BeanGPT backend.
"""

from typing import Optional


class BeanGPTException(Exception):
    """Base exception for all BeanGPT errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


class ConfigurationError(BeanGPTException):
    """Raised when there's a configuration issue."""
    pass


class ModelLoadingError(BeanGPTException):
    """Raised when model loading fails."""
    pass


class DatabaseError(BeanGPTException):
    """Raised when database operations fail."""
    pass


class APIError(BeanGPTException):
    """Raised when external API calls fail."""
    pass


class DataProcessingError(BeanGPTException):
    """Raised when data processing fails."""
    pass


class SecurityError(BeanGPTException):
    """Raised when security validation fails."""
    pass


class ValidationError(BeanGPTException):
    """Raised when input validation fails."""
    pass 