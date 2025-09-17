from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
import time
from contextlib import asynccontextmanager


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title = "Document Semantic Search Engine",
    description= "A semantic search engine for PDF, DOCX, and TXT documents using vector embeddings",
    version = "1.0.0"
)

# Configure CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:3000'], # React dev server
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

        # Create data directories
        os.makedirs("data/documents", exist_ok=True)
        os.makedirs("data/vectors", exist_ok=True)

        embedding_service = EmbeddingService()
        embedding_service.load_model()
        
        vector_store = FAISSVectorStore(
            dimension=embedding_service.get_embedding_dimension
        )

        vector_store.load_or_create_index()

        database_manager = DatabaseManager()

        logger.info("All services initialized successfully")

        yield
    
    except Exception as e:
        logger.error(f"Failed to load services: {str(e)}")
        raise e
    
@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Document Semantic Search Engine API",
        "status": "running",
        "total_documents": len(database_manager.get_all_documents()) if database_manager else 0,
        "total_vectors": vector_store.get_total_vectors() if vector_store else 0,
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    try:
        # Check if all services are available
        services_status = {
            "embedding_service": embedding_service is not None,
            "vector_store": vector_store is not None,
            "database_manager": database_manager is not None
        }

        return {
            "status": "healthy" if all(services_status.values()) else "unhealthy",
            "services": services_status,
            "total_documents": len(database_manager.get_all_documents()) if database_manager else 0,
            "total_vectors": vector_store.get_total_vectors() if vector_store else 0
        }
    except Exception as e:
        logger.error(f"Heatlh check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Heatlh check failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,host='0.0.0.0', port=8000)