from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
import os
import shutil
import uuid
from datetime import datetime
import logging
import json
from typing import List

from app.models.api_models import (
    DocumentUploadResponse,
    DocumentListResponse,
    DocumentMetadata,
    ProcessingStatus,
    ErrorResponse
)

from app.services.document_processor import DocumentProcessor
from app.dependencies import get_database_manager

logger = logging.getLogger(__name__)

router = APIRouter()

# Supported file extension and their MIME types
SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.txt'}
MIME_TYPE_MAP = {
    'application/pdf': '.pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx'
}

def validate_file(file:UploadFile)-> str:
    """
    Validate uploaded file and return file type
    Raises HTTPException if invalid
    """
    # Check file extension
    filename = file.filename.lower()
    file_ext = os.path.splitext(filename)[1]

    if file_ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Supported types: {','.join(SUPPORTED_EXTENSIONS)}"
        )
    
    # Check MIME type if available
    if file.content_type and file.content_type not in MIME_TYPE_MAP:
        # Allow text/plain variants
        if not file.content_type.startswith('text/') and file_ext != '.txt':
            raise HTTPException(
                status_code=400,
                detail=f"File content type {file.content_type} doesn't match extension {file_ext}"
            )
    
    # Check file size limit (limit to 50 mb)
    if hasattr(file, 'size') and file.size > 50 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File size too large. Max size: 50 MB"
        )
    
    return file_ext[1:] # Return extension without dot

def save_uploaded_file(file:UploadFile, file_type:str) -> tuple[str, int]:
    """Save uploaded file to disk and return (filename, file_size)"""
    # Generate unique filename
    file_id = str(uuid.uuid4())
    filename = f"{file_id}.{file_type}"
    file_path = os.path.join("data/documents", filename)

    # Ensure directory exists
    os.makedirs("data/documents", exist_ok=True)

    try:
        # Save file
        with open (file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Get file size
        file_size = os.path.getsize(file_path)

        return filename, file_size
    
    except Exception as e:
        # Cleanup on error
        if os.path.exists(file_path):
            os.unlink(file_path)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save file: {str(e)}"
        )
    
@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db_manager = Depends(get_database_manager)
):
    """
    Upload a document for processing
    """
    try:
        logger.info(f"Processing upload: {file.filename}")
        
        # Validate file
        file_type = validate_file(file)
        
        # Save file
        filename, file_size = save_uploaded_file(file, file_type)

        # Create database record
        document_id = db_manager.insert_document(
            filename = filename,
            original_filename = file.filename,
            file_type = file_type,
            file_size = file_size,
            metadata = json.dumps({"upload_timestamp": datetime.now().isoformat()})
        )

        # Queue background processing
        from app.dependencies import get_document_processor
        processor = get_document_processor()
        background_tasks.add_task(processor.process_document, document_id, filename)

        logger.info(f"Document uploaded successfully: {filename} (ID: {document_id})")

        return DocumentUploadResponse(
            document_id=document_id,
            filename=filename,
            original_filename=file.filename,
            file_size = file_size,
            processing_status = ProcessingStatus.PENDING,
            message = "Document uploaded successfully. Processing started."
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Upload failed: {str(e)}"
        )
    
@router.get("/", response_model=DocumentListResponse)
async def list_documents(db_manager = Depends(get_database_manager)):
    """Get list of all uploaded documents"""
    try:
        documents_data = db_manager.get_all_documents()

        documents = []
        for doc_data in documents_data:
            # Parse metadata if available
            metadata = None
            if doc_data.get('metadata'):
                try:
                    metadata = json.loads(doc_data['metadata'])
                except json.JSONDecodeError:
                    metadata = None
                    
            documents.append(DocumentMetadata(
                id = doc_data['id'],
                filename=doc_data['filename'],
                original_filename=doc_data['original_filename'],
                file_type=doc_data['file_type'],
                file_size=doc_data['file_size'],
                upload_timestamp=datetime.fromisoformat(doc_data['upload_timestamp']),
                processing_status=doc_data['processing_status'],
                num_chunks=doc_data['num_chunks'] or 0,
                metadata=metadata
            ))

        return DocumentListResponse(
            total_documents=len(documents_data),
            documents=documents
        )
        
    except Exception as e:
        logger.error(f"Failed to list documents: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve documents: {str(e)}"
        )

@router.get("/{document_id}", response_model=DocumentMetadata)
async def get_document(document_id:int, db_manager=Depends(get_database_manager)):
    """Get details of specific document"""
    try:
        documents = db_manager.get_all_documents()
        document_data = next((doc for doc in documents if doc['id'] == document_id), None)

        if not document_data:
            raise HTTPException(
                status_code=404,
                detail=f"Document with ID {document_id} not found"
            )
        
        # Parse metadata
        metadata = None
        if document_data.get('metadata'):
            try:
                metadata = json.loads(document_data['metadata'])
            except json.JSONDecodeError:
                metadata = None

        return DocumentMetadata(
            id = document_data['id'],
            filename=document_data['filename'],
            original_filename=document_data['original_filename'],
            file_type=document_data['file_type'],
            file_size=document_data['file_size'],
            upload_timestamp=datetime.fromisoformat(document_data['upload_timestamp']),
            processing_status=document_data['processing_status'],
            num_chunks=document_data['num_chunks'] or 0,
            metadata=metadata
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve document: {str(e)}"
        )
    
@router.delete("/{document_id}")
async def delete_document(document_id: int, db_manager = Depends(get_database_manager)):
    """
    Delete a document and associated data
    
    Note: This deletes files and database record.
    Vector store cleanup would require rebuiding the index.
    """
    try:
        documents = db_manager.get_all_documents()
        document_data = next((doc for doc in documents if doc['id'] == document_id), None)

        if not document_data:
            raise HTTPException(
                status_code=400,
                detail=f"Document with ID {document_id} not found"
            )
        
        file_path = os.path.join('data/documents', document_data['filename'])

        if os.path.exists(file_path):
            os.unlink(file_path)

        # Delete from database
        success = db_manager.delete_document(document_id)

        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to delete document from database."
            )

        return {
            "message": f"Document '{document_data['original_filaname']}' deleted successfully",
            "document_id": document_id
            }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document {document_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete document {document_id}: {str(e)}"
        )