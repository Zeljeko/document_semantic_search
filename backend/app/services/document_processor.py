import os
import logging
from typing import List, Dict, Any
import json
from datetime import datetime

from app.utils.document_parser import DocumentParser
from app.utils.text_chunker import TextChunker
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import FAISSVectorStore
from app.models.database import DatabaseManager

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Orchestrates complete document processing pipeline
    Parse -> Chunk -> Embed -> Store -> Index
    """

    def __init__(self,
                 embedding_service: EmbeddingService,
                 vector_store: FAISSVectorStore,
                 database_manager: DatabaseManager):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.database_manager = database_manager
        self.text_chunker = TextChunker(max_tokens=400, overlap_tokens=50)
    
    async def process_document(self, document_id: int, filename: str):
        """
        Process a single document through the pipeline

        """
        file_path = os.path.join("data/documents", filename)
        
        try:
            logger.info(f"Starting processing for document {document_id}: {filename}")

            # Update status to processing
            self.database_manager.update_document_status(document_id, status="processing")
            
            # -> Parse
            text, doc_metadata = DocumentParser.parse_document(file_path)
            logger.info(f"Extracted {len(text)} characters from {filename}")

            # -> Chunk
            chunks = self.text_chunker.chunk_document(text)
            logger.info(f"Created {len(chunks)} chunks from {filename}")

            if not chunks:
                    raise Exception("No chunks created from document")
            
            # -> Embed
            chunk_texts = [chunk['text'] for chunk in chunks]
            embeddings = self.embedding_service.generate_embeddings(chunk_texts)
            logger.info(f"Generated {len(embeddings)} embeddings")

            # -> Store
            # Prepping metadata
            vector_metadata = []
            for i, chunk in enumerate(chunks):
                 vector_metadata.append({
                      'document_id': document_id,
                      'chunk_id': chunk['chunk_id'],
                      'text': chunk['text'],
                      'token_count': chunk['token_count'],
                      'char_count': chunk['char_count'],
                      'filename': filename
                 })
            
            # Vector store
            self.vector_store.add_vectors(embeddings, vector_metadata)
            logger.info(f"Added {len(embeddings)} vectors to index")

            # Chunk metadata to database
            self.database_manager.insert_document_chunks(document_id, chunks)

            # Document status completed
            self.database_manager.update_document_status(document_id, "completed", len(chunks))

            # Save vector store to disk
            self.vector_store.save_index()
            logger.info(f"Successfully processed document {document_id}: {filename}")

        except Exception as e:
            logger.error(f"Failed to process document {document_id}: {filename}")

            # Update status to failed
            self.database_manager.update_document_status(document_id, "failed")
            logger.error(f"Document {document_id} processing failed: {str(e)}")
            raise e
        
    def get_processing_stats(self) -> Dict[Any, Any]:
        """Get statistics about document parsing"""
        try:
            documents = self.database_manager.get_all_documents()

            stats = {
                 'total_documents': len(documents),
                 'pending': len([d for d in documents if d['processing_status'] == 'pending']),
                 'processing': len([d for d in documents if d['processing_status'] == 'processing']),
                 'completed': len([d for d in documents if d['processing_status'] == 'completed']),
                 'failed': len([d for d in documents if d['processing_status'] == 'failed']),
                 'total_chunks': sum(d['num_chunks'] or 0 for d in documents),
                 'total_vectors': self.vector_store.get_total_vectors()
            }

            return stats
        
        except Exception as e:
            logger.error(f"Failed to get processing stats: {str(e)}")
            return {
             'error': str(e),
             'total_documents': 0,
             'total_vectors': 0    
            }
            