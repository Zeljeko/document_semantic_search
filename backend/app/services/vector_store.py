import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class FAISSVectorStore:
    """
    Manages FAISS vector index for similarity search
    Handles index creation, document storage, and similarity queries
    """

    def __init__(self, dimension:int, index_path: str = "data/vectors/faiss_index"):
        self.dimension = dimension
        self.index_path = index_path
        self.metadata_path = index_path + "_metadata.pkl"

        # FAISS index (will be created/loaded)
        self.index = None

        # Metadata store (maps vector indices to document info)
        self.metadata: List[Dict[Any, Any]] = []
        
        #Counter for Vector IDs
        self.next_id = 0

    def _create_index(self):
        """
        Create new FAISS index
        
        IndexFlatIP = Inner Product index (good for cosine similarity)
        """
        logger.info(f"Creating new FAISS index with dimension {self.dimension}")
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []
        self.next_id = 0

    def load_or_create_index(self):
        """Load existing index or create new one"""

        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self.load_index()
        else:
            self._create_index()

    def normalize_vectors(self, vectors:np.ndarray) -> np.ndarray:
        """
        Normalize vectors for cosine similarity
        -> Calculate L2 norm for each vector
        -> Divide each vector by its norm
        -> Return normalized vectors
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0,1,norms)
        return vectors/norms
    
    def add_vectors(self, vectors:np.ndarray, metadata:List[Dict[Any, Any]]):
        """
        Add vectors and their metadata to the index
        """
        if self.index is None:
            self.load_or_create_index()

        # Normalize vectors
        normalized_vectors = self.normalize_vectors(vectors)

        # Add to FAISS index
        self.index.add(normalized_vectors.astype('float32'))

        # Store metadata with IDs
        for meta in metadata:
            meta['vector_id'] = self.next_id
            self.next_id += 1
            self.metadata.append(meta)
        
        logger.info(f"Added {len(vectors)} vectors to index. Total: {self.index.ntotal}")

    def search(self, query_vector: np.ndarray, k: int = 10) -> List[Dict[Any, Any]]:
        """
        Search similar vectors
        -> Normalize query vector
        -> index.search() to find top k similar vectors
        -> Combine scores with metadata
        -> Return results sorted by similarity

        index.search() returns (scores, indices)
        """
        # Normalize query vector
        query_normalized = self.normalize_vectors(query_vector.reshape(1,-1))
        
        # Search FAISS index
        scores, indices = self.index.search(query_normalized.astype('float32'), k)

        # Combine results with metadata
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1: # -1 means no more result
                result = self.metadata[idx].copy()
                result['similarity_score'] = float(score)
                results.append(result)
        
        return results
    
    def save_index(self):
        """Save FAISS index and metadata to disk"""
        if self.index is None:
            return
        
        # Create directory if it doesnt exist
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, self.index_path)

        # Save metadata
        with open(self.metadata_path, 'wb') as f:
            pickle.dump({
                'metadata': self.metadata,
                'next_id': self.next_id
            }, f)
        
        logger.info(f"Index saved to {self.index_path}")

    def load_index(self):
        try:
            self.index = faiss.read_index(self.index_path)

            with open(self.metadata_path, 'rb') as f:
                data = pickle.load(f)
                self.metadata = data['metadata']
                self.next_id = data['next_id']

            logger.info(f"Index loaded from {self.index_path}. Total vectors: {self.index.ntotal}")
        
        except Exception as e:
            logger.error(f"Failed to load index: {str(e)}")
            self._create_index()
    
    def get_total_vectors(self) -> int:
        return self.index.ntotal if self.index else 0