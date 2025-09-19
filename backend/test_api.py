import requests
import json
import time
import os
from typing import Dict, Any

BASE_URL = "http://localhost:8000"
TEST_DOCUMENT_PATH = "test_document.txt"

class APITester:
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url,
        self.session = requests.Session()
        self.uploaded_document_id = None

    def create_test_document(self):
        test_content="""
Introduction
This is a comprehensive test document for the semantic search engine. It contains multiple paragraphs with different topics to test the chunking and search functionality.

Machine Learning and AI
Artificial intelligence and machine learning are transforming how we process and understand large amounts of text data. Vector embeddings allow us to capture semantic meaning in numerical form.

Vector Embeddings
Vector embeddings convert text into high-dimensional numerical representations. Similar texts have similar vector representations, enabling semantic search capabilities.

FAISS and Similarity Search
FAISS (Facebook AI Similarity Search) is a library for efficient similarity search. It can quickly find the most similar vectors in large datasets.

Python and FastAPI
This application is built using Python and FastAPI. FastAPI provides automatic API documentation and request validation.

Conclusion
This test document demonstrates various concepts related to document processing and semantic search functionality.
    """.strip()
        
        with open(TEST_DOCUMENT_PATH, 'w', encoding='utf-8') as f:
            f.write(test_content)

        print(f"Created test document: {TEST_DOCUMENT_PATH}")

    def test_health_check(self):
        print(f"Testing health check")
        
        response = self.session.get(f"{self.base_url}/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] in ["healthy","degraded"]
        assert "services" in data

        print(f"Health check passed: {data['status']}")
        return data
    
    def test_document_upload(self):
        print("Testing document upload")

        with open(TEST_DOCUMENT_PATH, 'rb') as f:
            files = {'file': (TEST_DOCUMENT_PATH, f, 'text/plain')}
            response = self.session.post(f"{self.base_url}/api/documents/upload", files=files)

        assert response.status_code == 200

        data = response.json()
        assert "document_id" in data
        assert data["processing_status"] == "pending"

        self.uploaded_document_id = data["document_id"]
        print(f"Document uploaded successfully: ID {self.uploaded_document_id}")
        return data
    
    def wait_for_processing(self, document_id:int, timeout:int = 60):
        print(f"Waiting for document {document_id} to process...")

        start_time = time.time()

        while time.time() - start_time < timeout:
            response = self.session.get(f"{self.base_url}/api/documents/{document_id}")
            assert response.status_code == 200

            data = response.json()
            status = data['processing_status']

            if status == "completed":
                print(f"Document processing completed in {time.time() - start_time:.1f}s")
                print(f"    Chunks created: {data['num_chunks']}")
                return data
            elif status == "failed":
                raise Exception(f"Document processing failed: {data}")
            
            print(f"    Status: {status}")
            time.sleep(2)

        raise Exception(f"Document processing timeout after {timeout}s")
    
    def test_document_list(self):
        print("Testing document listing")
        
        response = self.session.get(f"{self.base_url}/api/documents/")
        assert response.status_code == 200

        data = response.json()
        assert "documents" in data
        assert "total_documents" in data
        assert data["total_documents"] > 0

        print(f"Found {data['total_documents']} documents")
        return data
    
    def test_semantic_search(self):
        print("Testing semantic search")
    
        test_queries = [
            {
                "query": "What is machine learning?",
                "expected_topics": ["machine learning", "AI", "artificial intelligence"]
            },
            {
                "query": "How do vector embeddings work?",
                "expected_topics": ["vector", "embeddings", "representations"]
            },
            {
                "query": "Tell me about FAISS",
                "expected_topics": ["FAISS", "similarity search", "Facebook"]
            }
        ]

        for test_query in test_queries:
            print(f"    Query: '{test_query['query']}'")

            search_request = {
                "query": test_query["query"],
                "max_results": 5,
                "min_similarity": 0.0
            }

            response = self.session.post(
                f"{self.base_url}/api/search",
                json == search_request
            )
            assert response.status_code==200

            data = response.json()
            assert "results" in data
            assert "processing_time_ms" in data
            assert data["total_results"] > 0

            print(f"    Found {data['total_results']} results in {data['processing_time_ms']:.2f} ms")

            # Check if results are relevant
            top_result = data["results"][0]
            print(f"    Top result score: {top_result['similarity_score']}")
            print(f"    Top result preview: {top_result['text'][:100]}...")

        print("Semantic search tests passed")
        return True
    
    def test_search_stats(self):
        print("Testing search statistics")

        response = self.session.get(f"{self.base_url}/api/search/stats")
        assert self.response.status_code == 200

        data = response.json()
        assert "total_documents" in data
        assert "searchable_documents" in data
        assert "total_chunks" in data
        assert "total_vectors" in data
        
        print(f"Search stats: {data['searchable_documents']} docs, {data['total_chunks']} chunks")
        return data
    
    def test_error_handling(self):
        """Test API error handling"""
        print("Testing error handling")
        
        # Test invalid file upload
        files = {'file': ('test.xyz', b'invalid content', 'application/unknown')}
        response = self.session.post(f"{self.base_url}/api/documents/upload", files=files)
        assert response.status_code == 400
        print("   Invalid file type rejected")
        
        # Test invalid search query
        search_request = {"query": "", "max_results": 5}
        response = self.session.post(f"{self.base_url}/api/search/", json=search_request)
        assert response.status_code == 422
        print("   Empty search query rejected")
        
        # Test non-existent document
        response = self.session.get(f"{self.base_url}/api/documents/99999")
        assert response.status_code == 404
        print("   Non-existent document returns 404")
        
        print("Error handling tests passed")

    def cleanup(self):
        if os.path.exists(TEST_DOCUMENT_PATH):
            os.unlink(TEST_DOCUMENT_PATH)
            print(f"Cleaned up test file: {TEST_DOCUMENT_PATH}")
    
    def run_all_tests(self):
        print("=" * 60)
        print("DOCUMENT SEMANTIC SEARCH ENGINE - API TESTS")
        print("=" * 60)
        
        try:
            self.create_test_document()
            
            self.test_health_check()
            self.test_document_upload()
            
            if self.uploaded_document_id:
                self.wait_for_processing(self.uploaded_document_id)
                self.test_document_list()
                self.test_semantic_search()
                self.test_search_stats()
            
            self.test_error_handling()
            
            print("\n" + "=" * 60)
            print("ALL API TESTS PASSED!")
            print("Your semantic search engine is working correctly!")
            print("=" * 60)
            
        except Exception as e:
            print(f"TEST FAILED: {str(e)}")
            raise e
        
        finally:
            self.cleanup()

if __name__ == "__main__":

    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code != 200:
            print(f"Server not responding at {BASE_URL}")
            print("Start the server with: python app/main.py")
            exit(1)
    except requests.exceptions.ConnectionError:
        print(f"Cannot connect to server at {BASE_URL}")
        print("Start the server with: python app/main.py")
        exit(1)
    
    # Run tests
    tester = APITester()
    tester.run_all_tests()

            

