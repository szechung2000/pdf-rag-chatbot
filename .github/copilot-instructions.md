# PDF RAG Chatbot - AI Agent Instructions

## Architecture Overview

This is a **local-first RAG system** with three core components:
- **Ollama**: Local LLM inference (embedding + chat models)  
- **ChromaDB**: Vector database for semantic search
- **Streamlit**: Web interface for PDF upload and chat

Data flow: PDF → text extraction → chunking → embeddings → ChromaDB → retrieval + LLM generation.

## Key Development Patterns

### Environment & Dependencies
- **Container-first**: Everything runs in Docker, no local Python setup required
- **Model management**: Models auto-download on first startup via `docker-compose.yml` command chain
- **Volume strategy**: `data/`, `vectordb/`, `models/` are host-mounted for persistence

### Core File Responsibilities
- `app.py`: Single-file Streamlit app containing all logic (PDF processing, embeddings, chat)
- `docker-compose.yml`: Orchestrates Ollama service + app service with model auto-download
- `Dockerfile`: Python environment with PDF processing tools (poppler, tesseract)

### RAG Implementation Specifics
- **Chunking strategy**: Paragraph-aware chunking (800 chars, 100 overlap) with sentence fallback
- **Incremental indexing**: Tracks indexed PDFs by filename in ChromaDB metadata to avoid reprocessing
- **Embedding model**: `mxbai-embed-large:latest` via Ollama API calls
- **Chat model**: `gemma3n:e2b` with structured prompts including source citations

### State Management Patterns
```python
# Collection caching pattern used throughout
if "collection" not in st.session_state:
    st.session_state["collection"] = get_collection()

# Incremental indexing check
indexed_sources = get_indexed_sources()  # from ChromaDB metadata
new_pdfs = [pdf for pdf in pdf_files if os.path.basename(pdf) not in indexed_sources]
```

## Development Workflows

### Local Development
```bash
# Start full stack (first run downloads models - can take 10+ minutes)
docker-compose up

# Development with file watching (app.py changes auto-reload)
docker-compose restart rag-app

# Check model status
docker exec -it ollama ollama list
```

### Testing RAG Pipeline
- Use "Debug Info" expander in UI to test embeddings and view retrieved chunks
- Check model readiness with `check_models_available()` function
- Test incremental indexing by adding PDFs to `data/` folder

### Data Management
- PDFs: Place in `data/` directory (auto-detected by glob pattern)
- Vector DB: Persistent in `vectordb/` (ChromaDB SQLite format)
- Models: Cached in `models/` directory

## Integration Points

### Ollama API Communication
- **Health check**: `GET /api/tags` for model availability
- **Embeddings**: `POST /api/embeddings` with `{"model": EMBED_MODEL, "prompt": text}`
- **Generation**: `POST /api/generate` with structured prompt template
- **Host resolution**: `OLLAMA_HOST=http://ollama:11434` in Docker network

### ChromaDB Patterns
```python
# Standard collection pattern
client = chromadb.PersistentClient(path=VDB_DIR)
collection = client.get_or_create_collection("pdf_rag")

# Metadata-based source tracking
metadata = {"source": os.path.basename(pdf_path)}
```

### Error Handling Conventions
- Streamlit error display for API failures: `st.error(f"API Error {resp.status_code}: {resp.text}")`
- Graceful model unavailability with status messages
- Empty result handling in RAG pipeline

## Critical Configuration

### Model Requirements
- Embedding: `mxbai-embed-large:latest` (required for consistent vector dimensions)
- Chat: `gemma3n:e2b` (can be swapped, update `CHAT_MODEL` variable)
- GPU optional but recommended for performance

### Volume Persistence
- `vectordb/`: Must persist or embeddings will be regenerated
- `data/`: PDF source files  
- `ollama_data`: Model storage (several GB)

When modifying chunking, embedding models, or vector storage, always test incremental indexing behavior and ensure existing databases remain compatible.
