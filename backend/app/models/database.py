import sqlite3
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging 

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Manages SQLite database for document metadata.
    Stores document information, processing status and chunk metadata.
    """

    def __init__(self, db_path:str = "data/documents.db"):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Create tables if they don't exist"""
        # Create directory if needed
        dirpath = os.path.dirname(self.db_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Documents table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,      
                    filename TEXT NOT NULL,
                    original_filename TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    upload_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    processing_status TEXT DEFAULT 'pending',
                    num_chunks INTEGER DEFAULT 0,
                    metadata TEXT -- JSON string for additional metadata
                )
            ''')

            # Documents chunks table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL,
                    chunk_id INTEGER NOT NULL,
                    chunk_text TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    char_count INTEGER NOT NULL,
                    vector_id INTEGER, -- References FAISS index position
                    paragraph_indices TEXT, -- JSON string of paragraph indices
                    FOREIGN KEY (document_id) REFERENCES documents (id)
                )
            ''')

            conn.commit()
            logger.info("Database successfully initialized")

    def insert_document(self, filename: str, original_filename: str,
                        file_type: str, file_size: int, metadata: str = None) -> int:
        """Insert new document record and return document ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO documents
                (filename, original_filename, file_type, file_size, metadata)
                VALUES (?,?,?,?,?)
            ''', (filename, original_filename, file_type, file_size, metadata))
        
            document_id = cursor.lastrowid
            conn.commit()
            return document_id
    
    def insert_document_chunks(self,document_id:int,chunks:List[Dict[Any,Any]]):
        """Insert documents chunks for a document"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for chunk in chunks:
                cursor.execute('''
                INSERT INTO document_chunks
                (document_id, chunk_id, chunk_text, token_count, char_count, paragraph_indices)
                VALUES (?,?,?,?,?,?)
            ''',(
                document_id,
                chunk['chunk_id'],
                chunk['text'],
                chunk['token_count'],
                chunk['char_count'],
                str(chunk['paragraph_indices']) # store as string
            ))
        
            conn.commit()
    
    def update_document_status(self,document_id:int, status: str, num_chunks: int = None):
        """Update document processing status"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            if num_chunks is not None:
                cursor.execute('''
                    UPDATE documents
                    SET processing_status = ?, num_chunks = ?
                    WHERE id = ?               
                ''', (status, num_chunks, document_id))
            else:
                cursor.execute('''
                    UPDATE documents
                    SET processing_status = ?
                    WHERE id = ?
                ''', (status, document_id))

            conn.commit()

    def get_all_documents(self) -> List[Dict[Any, Any]]:
        """Get all documents and their metadata"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row # Return dict-like objects
            cursor = conn.cursor()

            cursor.execute('''
                SELECT * FROM documents
                ORDER BY upload_timestamp DESC               
            ''')

            return [dict(row) for row in cursor.fetchall()]
    
    def get_document_chunks(self,document_id: int) -> List[Dict[Any, Any]]:
        """Get all chunks for a specific document"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute('''
                SELECT * FROM document_chunks
                WHERE document_id = ?
                ORDER BY chunk_id                
            ''', (document_id,))

            return [dict(row) for row in cursor.fetchall()]