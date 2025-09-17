from fastapi import APIRouter, HTTPException, Depends, Query
import time
import logging
from typing import List

from app.models.api_models import SearchRequest, SearchResponse, DocumentChunkResponse
from app.main import get_embedding_service, get_vector_store, get_database_manager

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/", response_model=SearchResponse)
async def semantic_search(
    search_request: SearchRequest,
    embedding_service=Depends(get_embedding_service),
    vector_store=Depends(get_vector_store),
    db_manager=Depends(get_database_manager)
):
    """
    Perform semantic search on documents
    """
    start_time = time.time()
    
    try:
        logger.info(f"Processing search query: '{search_request.query}'")

        # Generate query embedding
        query_embedding = embedding_service.generate_single_embedding(search_request.query)

        # Search vector store
        raw_results = vector_store.search(
            query_embedding,
            k=search_request.max_results * 2 # Get extra results for filtering
        )

        # Filter by similarity threshold
        filtered_results = [
            result for result in raw_results
            if result['similarity_score'] >= search_request.min_similarity
        ]

        # Limit to max results
        filtered_results = filtered_results[:search_request.max_results]

        # Enrich with document metadata
        enriched_results = []
        documents_cache = {} # cache to avoid repeated database queries

        for result in filtered_results:
            document_id = result['document_id']

            # Get document metadata (cached)
            if document_id not in documents_cache:
                all_docs = db_manager.get_all_documents()
                doc_data = next((doc for doc in all_docs if doc['id'] == document_id), None)
                documents_cache[document_id] = doc_data
            
            doc_data = documents_cache[document_id]

            if doc_data:
                enriched_results.append(DocumentChunkResponse(
                    chunk_id = result['id'],
                    text = result['text'],
                    token_count = result['text'],
                    char_count = result['char_count'],
                    similarity_score = result['similarity_score'],
                    document_id = document_id,
                    document_filename = doc_data['original_filename']
                ))

            processing_time = (time.time() - start_time) * 1000 # convert to millisecond

            logger.info(f"Search completed: {len(enriched_results)} results in {processing_time:.2f}ms")

            return SearchResponse(
                query = search_request.query,
                total_results = len(enriched_results),
                max_results = search_request.max_results,
                processing_time_ms = round(processing_time, 2),
                results = enriched_results
            )
        
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )

@router.get("/suggestions")
async def get_search_suggestions(
    query: str = Query(..., min_length=1, max_length=100),
    limit: int = Query(default=5, ge=1, le=10),
):
    """
    Get search suggestions based on existing document content
    This could return common phrases or document titles that match the query
    For now, return a simple response
    """
    try:
        # Simplified implementation in production
        # Extract key phrases from all documents
        # Use fuzzy matching to find similar phrases
        # Return most relevant suggestions

        suggestions = [
            f"Search for: {query}",
            f"Documents containing: {query}",
            f"Similar to: {query}"
        ][:limit]

        return {
            "query": query,
            "suggestions": suggestions
        }
    
    except Exception as e:
        logger.error(f"Failed to get suggestions: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get suggestions: {str(e)}"
        )
    
@router.get("/stats")
async def get_search_stats(
    vector_store = Depends(get_vector_store),
    db_manager = Depends(get_database_manager)
):
    """Get statistics about the search index"""

    try:
        documents = db_manager.get_all_documents()
        completed_docs = [d for d in documents if d['processing_status'] == 'completed']
        
        stats = {
            "total_documents": len(documents),
            "searchable_documents": len(completed_docs),
            "total_chunks": sum(d['num_chunks'] or 0 for d in completed_docs),
            "total_vectors": vector_store.get_total_vectors(),
            "index_size_mb": "Not implemented",
            "average_chunks_per_document": (
                sum(d['num_chunks'] or 0 for d in completed_docs) / len(completed_docs)
                if completed_docs else 0
            )
        }

        return stats
    
    except Exception as e:
        logger.error(f"Failed to get stats: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get stats: {str(e)}"
        )
    