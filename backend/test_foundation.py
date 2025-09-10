#!/usr/bin/env python3
"""Test script for foundation components"""

import sys
import os
import tempfile
import shutil

# Add app to path
sys.path.append('app')

def test_document_parser():
    """Test document parsing functionality"""
    print("Testing Document Parser...")

    from app.utils.document_parser import DocumentParser

    # Create test TXT file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is a test document.\n\nIt has multiple paragraphs.\n\nThis is the third paragraph")
        test_file = f.name

    try:
        text,metadata = DocumentParser.parse_document(test_file)
        assert len(text) > 0
        assert metadata['file_type'] == 'txt'
        print("Document parser working correctly")
    finally:
        os.unlink(test_file)

def test_text_chunker():
    """Test text chunking functionality"""
    print("Testing Text Chunker...")

    from app.utils.text_chunker import TextChunker

    chunker = TextChunker(max_tokens=100, overlap_tokens=20)

    text = """This is the first paragraph of the test document. It contains multiple sentences to test chunking.
    
    This is the second paragraph. It should be a different chunk if the first paragraph is too long.
    
    This is the third paragraph. It tests overlap functionality between chunks.
    
    This is the fourth paragraph. It ensures our chunking algorithm works with multiple paragraphs."""

    chunks = chunker.chunk_document(text)

    assert len(chunks) > 0
    assert all('text' in chunk for chunk in chunks)
    assert all('token_count' in chunk for chunk in chunks)
    print(f"Text chunker created {len(chunks)} chunks")

    print(f"    First chunk: {chunks[0]['text'][:100]}...")

def test_embedding_service():
    """Test embedding generation"""
    print("Testing Embedding Service...")
    
    from app.services.embedding_service import EmbeddingService

    embedding_service = EmbeddingService()
    embedding_service.load_model()

    test_texts = ["This is a test sentence.", "This is another test sentence."]
    embeddings = embedding_service.generate_embeddings(test_texts)

    assert embeddings.shape[0] == 2 # Two sentences
    assert embeddings.shape[1] == 384 # Model dimension
    print(f"Embedding service generated {embeddings.shape} embeddings")

def test_vector_store():
    """Test FAISS vector store"""
    print("Testing Vector Store...")
    
    from app.services.vector_store import FAISSVectorStore
    from app.services.embedding_service import EmbeddingService

    embedding_service = EmbeddingService()
    embedding_service.load_model()
    
    vector_store = FAISSVectorStore(
        dimension=384,
        index_path="test_vectors/test_index"
    )
    vector_store.load_or_create_index()

    # Create test vector
    test_texts = [
        "Dogs are loyal animals.",
        "Cats are independent pets.",
        "Programming is fun and challenging."
    ]

    embeddings = embedding_service.generate_embeddings(test_texts)

    metadata = [
        {"text": text, "doc_id":i, "chunk_id":0}
        for i,text in enumerate(test_texts)
    ]

    # Add vectors
    vector_store.add_vectors(embeddings, metadata)

    # Test search
    query = "What are pets like?"
    query_embedding = embedding_service.generate_single_embedding(query)
    results = vector_store.search(query_embedding, k=2)

    assert len(results) > 0
    print(f"Vector store search returned {len(results)} results")
    print(f"    Top results: {results[0]['text'][:50]}... (score: {results[0]['similarity_score']:.3f})")

    # Cleanup
    shutil.rmtree("test_vectors", ignore_errors=True)

def test_database():
    """Test Database Operations"""
    print("Testing Database Manager...")

    from app.models.database import DatabaseManager

    # Use temporary database
    db_manager = DatabaseManager("test_db.sqlite")

    # Test document insertion
    doc_id = db_manager.insert_document(
        filename="test_doc.txt",
        original_filename="test_document.txt",
        file_type="txt",
        file_size=1024,
        metadata='{"test": true}'
    )

    assert doc_id > 0

    # Test chunk insertion
    test_chunks = [
        {
            "chunk_id":0,
            "text": "this is chunk 0",
            "token_count": 10,
            "char_count": 15,
            "paragraph_indices": [0]
        },
        {
            "chunk_id":1,
            "text": "this is chunk 1",
            "token_count": 12,
            "char_count": 15,
            "paragraph_indices": [1]
        }
    ]

    db_manager.insert_document_chunks(doc_id, test_chunks)
    db_manager.update_document_status(doc_id, "completed", len(test_chunks))

    # Test retrieval
    documents = db_manager.get_all_documents()
    chunks = db_manager.get_document_chunks(doc_id)

    assert len(documents) == 1
    assert len(chunks) == 2
    assert documents[0]['processing_status'] == 'completed'

    print(f"Database operations working correctly")
    print(f"    Created documents with {len(chunks)} chunks")

    # Cleanup
    os.unlink("test_db.sqlite")

def run_all_tests():
    """Run all foundation tests"""
    print("=" * 50)
    print("TESTING FOUNDATION COMPONENTS")
    print("=" * 50)

    try:
        test_document_parser()
        test_text_chunker()
        test_embedding_service()
        test_vector_store()
        test_database()

        print("\n" * 50)
        print("ALL FOUNDATIONS TEST PASSED")
        print("\n" * 50)
    
    except Exception as e:
        print("\n" * 50)
        print(f"TEST FAILED: {str(e)}")
        print("\n" * 50)
        raise e
    
if __name__ == "__main__":
    run_all_tests()