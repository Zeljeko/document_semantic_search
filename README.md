# Document Semantic Search API

A FastAPI-based backend that enables natural language search across PDF, DOCX, and TXT files. Instead of matching keywords, this system uses vector embeddings to understand the "meaning" behind queries.

Note: This project was built as a deep dive into vector math and backend architecture. Frontend development has been discontinued; this repository serves as a functional REST API.

### **Core Implementation**
I chose to avoid high-level wrappers like LangChain to better understand the underlying "plumbing" of AI applications.
- **Vector Engine**: FAISS (Facebook AI Similarity Search) for efficient similarity lookups.
- **Embeddings**: all-MiniLM-L6-v2 via Sentence-Transformers (384-dimensional vectors).
- **Database**: SQLite handles document metadata, while FAISS manages the vector index.
- **SProcessing**: Document is split into 400 token chunks with 50 token overlaps to maintain context across segments.

### **Tech Stack**

- **Language**: Designing end-to-end workflows from raw documents to searchable content
- **Framework**: Intelligent segmentation with context preservation through overlapping windows
- **Vector Store**: Hands-on experience with FAISS for production-scale similarity search
- **NLP**: Handling long-running AI operations without blocking user interactions
- **Parsing**: PyPDF2, python-docx 

| Component  | Technology |
| ------------- | ------------- |
| Language  | Python 3.13.5 |
| Framework  | FastAPI (Pydantic 2 for validation) |
| Vector Store | FAISS |
| NLP | Sentence-Transfomers (PyTorch) |
| Parsing | PyPDF2, python-docx |

### **How it works**

```
┌─────────────────────────────────────────────────────────────┐
│                     DOCUMENT UPLOAD                          │
│                          ↓                                   │
│  ┌──────────────────────────────────────────────────┐       │
│  │  Document Parser (PDF/DOCX/TXT)                  │       │
│  └──────────────────────┬───────────────────────────┘       │
│                         ↓                                    │
│  ┌──────────────────────────────────────────────────┐       │
│  │  Text Chunker (Paragraph-based, 50-token overlap)│       │
│  └──────────────────────┬───────────────────────────┘       │
│                         ↓                                    │
│  ┌──────────────────────────────────────────────────┐       │
│  │  Embedding Service (sentence-transformers)       │       │
│  └──────────────────────┬───────────────────────────┘       │
│                         ↓                                    │
│  ┌─────────────────────────────┬────────────────────┐       │
│  │  FAISS Vector Store         │  SQLite Database   │       │
│  │  (Similarity Search)        │  (Metadata)        │       │
│  └─────────────────────────────┴────────────────────┘       │
│                         ↓                                    │
│  ┌──────────────────────────────────────────────────┐       │
│  │  Search API (Query → Results with Scores)        │       │
│  └──────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────┘
```
1. **Document Upload** → User uploads PDF, DOCX, or TXT file
2. **Validation** → File type, size, and format verification
3. **Text Extraction** → Format-specific parsing to clean text
4. **Intelligent Chunking** → Split into semantic units with overlap
5. **Embedding Generation** → Convert chunks to 384-dimensional vectors
6. **Vector Indexing** → Store in FAISS for fast similarity search
7. **Metadata Storage** → Track documents and chunks in SQLite

### Quick Start
## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd document-semantic-search

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install backend dependencies
cd backend
pip install -r requirements.txt

# Create data directories
mkdir -p data/documents data/vectors
```

## Running the Backend

```bash
# Start the FastAPI server
cd backend
python app/main.py

# Server will start on http://localhost:8000
# API documentation available at http://localhost:8000/docs
```

## Testing the API

```bash
# Health check
curl http://localhost:8000/health

# Upload a document
curl -X POST http://localhost:8000/api/documents/upload \
  -F "file=@path/to/your/document.pdf"

# Search documents
curl -X POST http://localhost:8000/api/search/ \
  -H "Content-Type: application/json" \
  -d '{"query": "What is machine learning?", "max_results": 5}'

# List all documents
curl http://localhost:8000/api/documents/
```

## Testing

```bash
# Run foundation tests
cd backend
python test_foundation.py

# Run API integration tests
python test_api.py

# Expected output: All tests passing with performance metrics
```

## Known Limitations

- **Single-user**: Currently no authentication
- **Local storage**: Data stored locally, no cloud integration
- **Processing queue**: Sequential document processing (no parallel workers)
- **Vector compression**: Using exact search (IndexFlatIP), not optimized for very large datasets
