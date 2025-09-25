from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import logging
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)

class DocumentProcessingError(Exception):
    """Custom exception for document processing errors"""
    def __init__(self, message: str, document_id: int = None):
        self.message = message
        self.document_id = document_id
        super().__init__(self.message)

class VectorStoreError(Exception):
    """Custom exception for vector store errors"""
    def __init__(self, message: str, document_id: int = None):
        self.message = message
        self.document_id = document_id
        super().__init__(self.message)

async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle request validation error with detailed feedback
    """
    errors = []
    for error in exc.errors():
        field = " -> ".join(str(x) for x in error["loc"])
        message = error["msg"]
        errors.append(f"{field}: {message}")

    error_detail = {
        "error": "Validation Error",
        "detail": f"Invalid request: {', '.join(errors)}",
        "timestamp": datetime.now().isoformat()
    }

    logger.warning(f"Validation error: {error_detail}")
    return JSONResponse(status_code=422, content=error_detail)

async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Handle HTTP exceptions with consistent error format
    """
    error_detail = {
        "error": f"HTTP {exc.status_code}",
        "detail": exc.detail,
        "timestamp": datetime.now().isoformat()
    }

    logger.error(f"HTTP exception: {error_detail}")
    return JSONResponse(status_code=exc.status_code, content=error_detail)

async def general_exception_handler(request: Request, exc: Exception):
    """
    Handle unexpected exceptions
    """
    error_detail = {
        "error": "Internal Server Error",
        "detail": "An unexpected error occured. Please try again later.",
        "timestamp": datetime.now().isoformat()
    }

    logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
    return JSONResponse(status_code=500, content=error_detail)

def validate_search_query(query: str) -> str:
    """
    Validate and clean search query
    """
    if not query or not query.strip():
        raise HTTPException(
            status_code=400,
            detail="Search query cannot be empty"
        )
    
    query = query.strip()

    if len(query) > 500:
        raise HTTPException(
            status_code=400,
            detail=f"Search query too long. Maximum 500 characters."
        )
    
    # Remove potentially problematic characters

    cleaned_query = "".join(c for c in query if c.isprintable())

    return cleaned_query

def validate_file_upload(file_size: int, filename: str) -> None:
    """
    Validate file upload parameters
    """ 
    MAX_FILE_SIZE = 50 * 1024 * 1024 # 50MB

    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File size too large: {file_size/(1024 * 1024):.1f}MB. Maximum: 50MB"
        )
    
    if not filename or len(filename.strip()) == 0:
        raise HTTPException(
            status_code=400,
            detail="Filename cannot be empty"
        )
    
    # Check for potentially dangerous filenames
    dangerous_chars = ['..', '/', '\\', '<', '>', ':', '"', '|', '?', '*']
    if any(char in filename for char in dangerous_chars):
        raise HTTPException(
            status_code=400,
            detail="Filename contains invalid characters"
        )