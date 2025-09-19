from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
import logging
import os
import time
from datetime import datetime
from contextlib import asynccontextmanager

# Import error handlers
from app.utils.error_handlers import (
    validation_exception_handler,
    http_exception_handler,
    general_exception_handler
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title = "Document Semantic Search Engine",
    description= """
    A semantic search engine for documents using vector embeddings

    ## Features
    * Upload PDF, DOCX, TXT documents
    * Automatic text chunking and embedding generation
    * FAISS powered similarity search
    * RESTful API with comprehensive error handling

    ## Workflow
    1. **Upload** documents via '/api/documents/upload
    2. **Wait** for processing to complete (check status via 'api/documnets/')
    3. **Search** using '/api/search/' with natural language queries
    """,
    version = "1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",  
)

app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(HTTPException, http_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:3000', 'http://127.0.0.1:3000'], # React dev server
    allow_credentials = True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Global services (will be initialized on startup)
embedding_service = None
vector_store = None
database_manager = None
document_processor = None
start_time = time.time()

# Dependency injection functions
def get_embedding_service():
    if embedding_service is None:
        raise HTTPException(status_code=503, detail="Embedding service is not initialized")
    return embedding_service

def get_vector_store():
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store is not initialized")
    return vector_store

def get_database_manager():
    if database_manager is None:
        raise HTTPException(status_code=503, detail="Database manager is not initialized")
    return database_manager

def get_document_processor():
    if document_processor is None:
        raise HTTPException(status_code=503, detail="Document processor is not initialized")
    return document_processor

# Lifespan of application
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global embedding_service, vector_store, database_manager

    logger.info("Initializing Document Semantic Search Engine...")
    
    try:
        # Initialize services
        from app.services.embedding_service import EmbeddingService
        from app.services.vector_store import FAISSVectorStore
        from app.models.database import DatabaseManager
        from app.services.document_processor import DocumentProcessor

        # Create data directories
        os.makedirs("data/documents", exist_ok=True)
        os.makedirs("data/vectors", exist_ok=True)
        
        logger.info("Initializing embedding service...")
        embedding_service = EmbeddingService()
        embedding_service.load_model()
        logger.info("Embedding service ready.")
        
        logger.info("Initializing vector store...")
        vector_store = FAISSVectorStore(
            dimension=embedding_service.get_embedding_dimension
        )

        vector_store.load_or_create_index()
        logger.info("Vector store ready.")
        
        logger.info("Initializing database...")
        database_manager = DatabaseManager()
        logger.info("Database ready.")

        logger.info("Initializing document processor...")
        document_processor = DocumentProcessor(
            embedding_service=embedding_service,
            vector_store=vector_store,
            database_manager=database_manager
        )
        logger.info("Document processor ready.")

        # Log system stats
        from app.utils.performance import PerformanceOptimizer

        recommendations = PerformanceOptimizer.get_processing_recommendation()
        logger.info(f"Performance recommendations: {recommendations}")
        
        logger.info("All services initialized successfully.")

        yield
    
    except Exception as e:
        logger.error(f"Failed to load services: {str(e)}")
        raise e
    
@app.get("/")
async def root():
    """API root with system status"""
    uptime = time.time() - start_time
    return {
        "message": "Document Semantic Search Engine API",
        "status": "running",
        "version": "1.0.0",
        "uptime_seconds": round(uptime,2),
        "total_documents": len(database_manager.get_all_documents()) if database_manager else 0,
        "total_vectors": vector_store.get_total_vectors() if vector_store else 0,
        "docs_url": "/docs",
        "redoc_url": "/redoc"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        uptime = time.time() - start_time
        # Check if all services are available
        services_status = {
            "embedding_service": embedding_service is not None,
            "vector_store": vector_store is not None,
            "database_manager": database_manager is not None,
            "document_processor": document_processor is not None
        }

        # Get processing stats if available
        processing_stats = {}
        if document_processor:
            try: 
                processing_stats = document_processor.get_processing_stats()
            except Exception as e:
                processing_stats = {"error": str(e)}

        return {
            "status": "healthy" if all(services_status.values()) else "degraded",
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": round(uptime, 2), 
            "services": services_status,
            "processing_stats": processing_stats
        }
    except Exception as e:
        logger.error(f"Heatlh check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Heatlh check failed: {str(e)}")

# Include API Routes
from app.api import documents, search

app.include_router(
    documents.router,
    prefix="/api/documents",
    tags = ["Document Management"]
)

app.include_router(
    search.router,
    prefix="/api/search",
    tags = ["Semantic Search"]
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,    
                host='0.0.0.0', 
                port=8000,
                log_level="info",
                reload=False # Set to true for development
    )