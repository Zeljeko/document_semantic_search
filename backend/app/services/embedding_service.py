import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class EmbeddingService:
    """
    Handles text to vector embedding operations using sentence-transformers.
    Manages model loading and embedding generation.
    """

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initialize embedding service with specific model
        """
        self.model_name=model_name
        self.model=None
        self.embedding_dimension = 384
    
    def load_model(self):
        """Load sentence transformers model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {self.model_name}")
            raise Exception(f"Could not load embedding model: {str(e)}")
        
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embedding for lists of text
        """
        if self.model is None:
            self.load_model()
        
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise Exception(f"Embedding generation failed: {str(e)}")
        
    def generate_single_embedding(self, text: str) -> np.ndarray:
        return self.generate_embeddings([text])[0]
    
    def get_embedding_dimension(self) -> int:
        return self.embedding_dimension