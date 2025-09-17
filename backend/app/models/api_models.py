from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class ProcessingStatus(str, Enum):
    """Document processing status options"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class FileType(str, Enum):
    """Supported file types"""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"

# Request models
class SearchRequest(BaseModel):
    """Request Model for semantic search"""
    query:str = Field(
        min_length=1,
        max_length=500,
        description="Search query text"
    )
    max_results: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of results to return"
    )
    min_similarity:float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold (0.0 to 1.0)"
    )

    @field_validator('query')
    def validate_query(cls,v):
        if not v.strip():
            raise ValueError('Query cannot be empty or whitespace only')
        return v.strip()
    
# Response models
class DocumentChunkResponse(BaseModel):
    """Response model for document chunk information"""
    chunk_id: int
    text: str
    token_count: int
    char_count: int
    similarity_score: float
    document_id: int
    document_filename: str

class SearchResponse(BaseModel):
    """Response model for search results"""
    query: str
    total_results: int
    max_results: int
    processing_time_ms: float
    results: List[DocumentChunkResponse]
    
class DocumentMetadata(BaseModel):
    """Document metadata for responses"""
    id: int
    filename: str
    original_filename: str
    file_type: FileType
    file_size: int
    upload_timestamp: datetime
    processing_status: ProcessingStatus
    num_chunks: int
    metadata: Optional[Dict[str, Any]] = None

class DocumentListResponse(BaseModel):
    """Response model for document listing"""
    total_documents: int
    documents: List[DocumentMetadata]

class DocumentUploadResponse(BaseModel):
    """Response model for document upload"""
    document_id: int
    filename: str
    original_filename: str
    file_size: str
    processing_status: ProcessingStatus
    message: str

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    services: Dict[str, bool]
    total_document: int
    total_vectors: int
    uptime_seconds: Optional[float] = None

class ErrorResponse(BaseModel):
    """Standard error response model"""
    error: str
    detail: str
    timestamp: datetime = Field(default_factory=datetime.now())