import re
from typing import List
from sentence_transformers import SentenceTransformer

class TextChunker:
    """
    Handles intelligent text chunking for vector embeddings.
    Splits texts into semantic chunks with overlap.
    """

    def __init__(self, max_tokens:int = 400, overlap_tokens: int = 50):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.tokenizer = SentenceTransformer('all-MiniLM-L6-v2')

    def count_tokens(self, text:str) -> int:
        """Count tokens using sentence-transfomers tokenizer"""
        tokens = self.tokenizer.tokenizer.tokenize(text)
        return len(tokens)
    
    def split_by_paragraphs(self, text:str) -> List[str]:
        """
        Cleanly separates paragraphs by the regex pattern:
        new line -> any repeating optional whitespace -> new line
        Each separated paragraph is sanitized in list
        Returns separated paragraphs
        """
        paragraphs = re.split(r'\n\s*\n', text.strip())
        return [p.strip() for p in paragraphs if p.strip()]
    
    def create_chunks_with_overlap(self, paragraphs: List[str]) -> List[dict]:
        """
        Create chunks from paragraphs with token-aware overlap
        Returns list of chunk dicionaries with metadata
        """
        chunks = []
        current_chunk = ""
        chunk_paragraphs = []
        
        for i, paragraph in enumerate(paragraphs):
            # Test adding this paragraph
            test_chunk = current_chunk + "\n" + paragraph if current_chunk else paragraph
            token_count = self.count_tokens(test_chunk)

            if token_count <= self.max_tokens:
                # Add paragraph to current chunk
                current_chunk = test_chunk
                chunk_paragraphs.append(i)
            else:
                # Current chunk is full, save it and start a new one
                if current_chunk:
                    chunks.append({
                        "text": current_chunk,
                        "token_count": self.count_tokens(current_chunk),
                        "paragraph_indices": chunk_paragraphs.copy(),
                        "chunk_id": len(chunks)
                    })

                # Start new chunk with overlap
                overlap_text = self._create_overlap(current_chunk)
                current_chunk = overlap_text + "\n" + paragraph if overlap_text else paragraph
                chunk_paragraphs = [i] # New chunk starts with current paragraph
        
        # Don't forget last chunk
        if current_chunk:
            chunks.append({
                "text": current_chunk,
                "token_count": self.count_tokens(current_chunk),
                "paragraph_indices": chunk_paragraphs,
                "chunk_id": len(chunks)
            })
        
        return chunks

    def _create_overlap(self,text:str) -> str:
        """Create overlap text from end of previous chunk"""
        if not text:
            return ""
        
        # Get the last part that fits in the overlap token limit
        sentences = re.split(r'[.!?]+', text)
        overlap = ""

        # Build overlap from last sentences
        for sentence in reversed(sentences):
            test_overlap = sentence.strip() + ". " + overlap if overlap else sentence.strip()
            if self.count_tokens(test_overlap) <= self.overlap_tokens:
                overlap = test_overlap
            else:
                break

        return overlap.strip()
    
    def chunk_document(self, text:str) -> List[dict]:
        """
        Main chunking method
        Splits text into paragraph then create chunks with overlap 
        Add metadata to each chunk then return processed chunks
        """
        paragraphs = self.split_by_paragraphs(text)
        chunks = self.create_chunks_with_overlap(paragraphs)

        # Add additional metadata
        for chunk in chunks:
            chunk["char_count"] = len(chunk["text"])
            chunk["has_overlap"] = chunk["chunk_id"]>0
        
        return chunks

