# Document Semantic Search Engine

> **Work in Progress** - Currently implementing frontend interface (React + Tailwind CSS)

A semantic search engine built from scratch to understand and apply modern AI engineering practices. Unlike keyword-based search, this system uses vector embeddings to understand the semantic meaning of text, enabling natural language queries to find relevant content even when exact keywords don't match.

**Project Goal**: Build a production-oriented AI application that demonstrates practical integration of large language models, vector databases, and modern software engineering practices—moving beyond tutorials to solve real-world document search challenges.

## Learning Objectives & Skills Demonstrated

This project serves as a hands-on exploration of key AI engineering competencies:

### **Core AI/ML Concepts**

- **Vector Embeddings**: Understanding how text is converted into numerical representations that capture semantic meaning
- **Similarity Search**: Implementing cosine similarity for finding semantically related content
- **Embedding Models**: Working with sentence-transformers and understanding model selection trade-offs
- **Semantic vs. Keyword Search**: Practical differences and when to use each approach

### **Production AI Engineering**

- **Document Processing Pipelines**: Designing end-to-end workflows from raw documents to searchable content
- **Text Chunking Strategies**: Intelligent segmentation with context preservation through overlapping windows
- **Vector Databases**: Hands-on experience with FAISS for production-scale similarity search
- **Asynchronous Processing**: Handling long-running AI operations without blocking user interactions

### **Software Engineering Practices**

- **API Design**: RESTful endpoints with FastAPI, automatic documentation, and proper validation
- **Error Handling**: Comprehensive exception handling and user-friendly error messages
- **Database Design**: SQLite schema for metadata management and document tracking
- **Testing**: Unit tests, integration tests, and system validation
- **Code Organization**: Service-oriented architecture with clear separation of concerns

### **DevOps & Deployment** _(Upcoming)_

- Docker containerization and multi-service orchestration
- Production configuration and environment management
- Monitoring, logging, and observability

## Current Features

### Completed (Backend)

- **Multi-Format Document Upload**: Support for PDF, DOCX, and TXT files
- **Intelligent Document Processing**:
  - Automatic text extraction from various formats
  - Paragraph-based chunking with 50-token overlap to preserve context
  - Token-aware segmentation (max 400 tokens per chunk)
- **Vector Embedding Generation**: Using sentence-transformers (all-MiniLM-L6-v2)
- **FAISS Vector Store Integration**: Production-ready similarity search with normalized cosine similarity
- **Semantic Search API**: Natural language queries with configurable relevance thresholds
- **Comprehensive Error Handling**: Robust validation and meaningful error messages
- **API Documentation**: Automatic OpenAPI/Swagger documentation
- **Background Processing**: Non-blocking document processing with status tracking

### In Development

- **React Frontend**: Interactive UI with drag-and-drop file upload
- **Real-time Status Updates**: Live progress tracking for document processing
- **Search Interface**: Clean search bar with filters and result ranking
- **Docker Deployment**: Complete containerization setup

### Planned Features

- Advanced search filters (date range, document type, similarity thresholds)
- Result highlighting and text snippets
- Document summarization
- Performance optimization for large document collections

## Technology Stack

### Backend (Current Focus)

- **FastAPI**: Modern Python web framework with automatic API documentation
- **Sentence Transformers**: Pre-trained models for generating semantic embeddings
- **FAISS**: Facebook's library for efficient similarity search at scale
- **SQLite**: Lightweight database for document metadata
- **PyPDF2 & python-docx**: Document parsing libraries
- **Pydantic**: Data validation and settings management

### Frontend (Next Phase)

- **React**: Component-based UI library
- **Tailwind CSS**: Utility-first CSS framework
- **Axios**: HTTP client for API communication

### Deployment (Future)

- **Docker & Docker Compose**: Containerization
- **Nginx**: Reverse proxy and load balancing

## 🏗️ System Architecture

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

### Processing Pipeline

1. **Document Upload** → User uploads PDF, DOCX, or TXT file
2. **Validation** → File type, size, and format verification
3. **Text Extraction** → Format-specific parsing to clean text
4. **Intelligent Chunking** → Split into semantic units with overlap
5. **Embedding Generation** → Convert chunks to 384-dimensional vectors
6. **Vector Indexing** → Store in FAISS for fast similarity search
7. **Metadata Storage** → Track documents and chunks in SQLite

### Search Flow

1. **Query Input** → User enters natural language query
2. **Query Embedding** → Convert query to vector using same model
3. **Similarity Search** → FAISS finds most similar chunk vectors
4. **Ranking** → Sort by cosine similarity score
5. **Result Enrichment** → Add document metadata and context
6. **Response** → Return ranked results with relevance scores

## Quick Start

### Prerequisites

- Python 3.9 or higher
- pip and virtual environment tools
- ~2GB RAM for embedding model and vector operations
- ~200MB disk space for model cache

### Installation

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

### Running the Backend

```bash
# Start the FastAPI server
cd backend
python app/main.py

# Server will start on http://localhost:8000
# API documentation available at http://localhost:8000/docs
```

### Testing the API

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

## 📊 Current Performance

- **Search Latency**: < 100ms for datasets under 1,000 chunks
- **Upload Processing**: 2-5 seconds per document (varies by size and format)
- **Supported Scale**: Tested with 100+ documents and 1,000+ chunks
- **Memory Usage**: ~1.5GB for 100 documents with embeddings loaded

## 🧪 Testing

```bash
# Run foundation tests
cd backend
python test_foundation.py

# Run API integration tests
python test_api.py

# Expected output: All tests passing with performance metrics
```

## API Documentation

### Interactive Documentation

Once the server is running, visit:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Key Endpoints

#### Document Management

- `POST /api/documents/upload` - Upload a document for processing
- `GET /api/documents/` - List all uploaded documents with status
- `GET /api/documents/{id}` - Get details of a specific document
- `DELETE /api/documents/{id}` - Delete a document and its data

#### Semantic Search

- `POST /api/search/` - Perform semantic search with natural language query
  - Parameters: `query` (string), `max_results` (int), `min_similarity` (float)
- `GET /api/search/stats` - Get search index statistics

#### System Health

- `GET /health` - System health check with service status
- `GET /` - API root with system information

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the `backend/` directory:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# File Upload
MAX_FILE_SIZE=52428800  # 50MB

# Embedding Model
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Vector Store
VECTOR_STORE_PATH=data/vectors/faiss_index

# Database
DATABASE_PATH=data/documents.db
```

## 🎓 Key Technical Decisions & Learnings

### Why Sentence-Transformers?

- Pre-trained models optimized for semantic similarity tasks
- Good balance of quality and performance for document search
- all-MiniLM-L6-v2 chosen for: 384 dimensions, 80MB size, fast inference

### Why FAISS?

- Industry-standard vector database from Facebook AI Research
- Efficient similarity search with multiple index types
- Scales from thousands to billions of vectors
- Can transition from exact (IndexFlatIP) to approximate search as data grows

### Chunking Strategy

- Paragraph-based chunking preserves semantic coherence
- 50-token overlap prevents context loss at boundaries
- 400-token max keeps chunks within embedding model limits
- Balances search precision with processing efficiency

### Why FastAPI?

- Automatic OpenAPI documentation generation
- Built-in data validation with Pydantic
- Excellent async support for I/O-bound operations
- Modern Python with type hints throughout

## Known Issues & Limitations

- **Single-user**: Currently no authentication or multi-tenancy support
- **Local storage**: Data stored locally, no cloud integration yet
- **Processing queue**: Sequential document processing (no parallel workers)
- **Vector compression**: Using exact search (IndexFlatIP), not optimized for very large datasets
- **No frontend**: Command-line/API testing only until React interface is complete

## 🔜 Next Steps

### Immediate

- [ ] Build React frontend with file upload interface
- [ ] Implement real-time progress tracking
- [ ] Create search UI with result display
- [ ] Add Tailwind CSS styling

### Near Future

- [ ] Docker containerization
- [ ] Frontend-backend integration
- [ ] End-to-end testing
- [ ] Deployment preparation

### Future Enhancements

- [ ] User authentication and access control
- [ ] Advanced search filters and query suggestions
- [ ] Document summarization and highlighting
- [ ] Support for additional document formats
- [ ] Distributed vector storage for scalability
- [ ] Question-answering capabilities

## Project Structure

```
document-semantic-search/
├── backend/
│   ├── app/
│   │   ├── api/              # API route handlers
│   │   ├── models/           # Data models and database
│   │   ├── services/         # Business logic (embeddings, vectors)
│   │   └── utils/            # Document parsing, chunking, errors
│   ├── data/                 # Document storage and vector indices
│   ├── requirements.txt      # Python dependencies
│   └── test_*.py            # Test files
├── frontend/                 # (Coming soon)
└── README.md
```

## 🤝 Learning Resources Used

- [Sentence Transformers Documentation](https://www.sbert.net/)
- [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Vector Embeddings Guide](https://www.pinecone.io/learn/vector-embeddings/)

## Development Notes

This project intentionally avoids using high-level abstractions or managed services (like LangChain or Pinecone) in favor of implementing core concepts from scratch. The goal is to develop deep understanding of:

- How vector embeddings work fundamentally
- Vector database operations and optimization
- Document processing pipeline design
- Production API patterns and error handling

The hands-on implementation approach provides stronger foundational knowledge applicable across different AI/ML tools and frameworks.

## License

MIT License - Feel free to use this project for learning purposes.

---

**Built with the goal of understanding AI engineering from first principles**

**Status**: Backend complete | Frontend in progress | Deployment upcoming
