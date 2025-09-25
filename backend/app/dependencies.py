from fastapi import HTTPException

# Global services (will be initialized on startup)
embedding_service = None
vector_store = None
database_manager = None
document_processor = None

def set_services(e_s, v_s, db_m, d_p):
    """Called by main.py during startup to set the services"""
    global embedding_service, vector_store, database_manager, document_processor
    embedding_service = e_s
    vector_store = v_s
    database_manager = db_m
    document_processor = d_p
    
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