# Local PDF RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that allows you to chat with your PDF documents using local models via Ollama.

## Features

- 📄 Extract text from PDF documents
- 🔍 Semantic search using embeddings
- 💬 Chat with your documents using local LLMs
- 🗂️ ChromaDB vector database for efficient storage
- 🔄 Incremental indexing (only processes new PDFs)
- 🐳 Full Docker setup with Ollama included

## Requirements

- Docker and Docker Compose

## Quick Start

1. **Clone this repository:**
   ```bash
   git clone https://github.com/yourusername/pdf-rag-chatbot.git
   cd pdf-rag-chatbot
   ```

2. **Add your PDF files:**
   ```bash
   # Copy your PDF files to the data/ directory
   mkdir -p data
   cp /path/to/your/pdfs/*.pdf data/
   ```

3. **Start the application:**
   ```bash
   docker-compose up
   ```

4. **Wait for models to download:**
   The first run will download the required models:
   - `mxbai-embed-large` (embedding model)
   - `gemma3n:e2b` (chat model)
   
   Watch the logs for completion messages:
   ```
   ✅ mxbai-embed-large completed!
   ✅ gemma3n:e2b completed!
   🎉 All models ready!
   ```

5. **Access the application:**
   - Open your browser to http://localhost:8501

## Usage

1. **Index your PDFs:**
   - Click "🔄 Index all PDFs" to process your documents

2. **Start chatting:**
   - Type questions about your PDFs in the text input
   - The system will find relevant content and generate answers

## Configuration

You can modify these variables in `app.py`:

- `EMBED_MODEL`: Embedding model for semantic search
- `CHAT_MODEL`: Chat model for generating responses  
- `DATA_DIR`: Directory containing PDF files (mounted to `/app/data`)
- `VDB_DIR`: Vector database storage directory (mounted to `/app/vectordb`)

## Project Structure

```
├── app.py              # Main Streamlit application
├── Dockerfile          # Container configuration
├── docker-compose.yml  # Multi-service orchestration
├── requirements.txt    # Python dependencies (optional)
├── data/               # Place your PDF files here (mounted volume)
├── vectordb/           # ChromaDB storage (mounted volume, auto-created)
├── models/             # Model cache (mounted volume, auto-created)
└── README.md          # This file
```

## Services

The docker-compose setup includes:

- **ollama**: Runs the Ollama server with model management
- **rag-app**: Streamlit application for the RAG interface

## Volumes

- `data/`: Your PDF files (persistent)
- `vectordb/`: ChromaDB embeddings storage (persistent)
- `models/`: Model cache (persistent)
- `ollama_data`: Ollama internal data (persistent)

## Ports

- `8501`: Streamlit web interface
- `11434`: Ollama API (internal)

## Troubleshooting

- **Models not downloading**: Check Docker logs with `docker-compose logs ollama`
- **App can't connect to Ollama**: Ensure both services are in the same network
- **PDFs not found**: Make sure PDF files are in the `data/` directory
- **Embeddings not persisting**: Check that `vectordb/` directory has write permissions
- **Out of memory**: Large models require significant RAM (8GB+ recommended)

## Commands

```bash
# Start services
docker-compose up

# Start in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Rebuild containers
docker-compose up --build

# Clean everything (including volumes)
docker-compose down -v
```

## Manual Model Management

If you need to manage models manually:

```bash
# Access Ollama container
docker exec -it ollama bash

# List available models
ollama list

# Pull additional models
ollama pull llama2

# Remove models
ollama rm model_name
```

## Development

To modify the application:

1. Edit `app.py` or other files
2. Restart the container: `docker-compose restart rag-app`
3. Changes are reflected immediately due to volume mounting

## Performance Notes

- First startup takes longer due to model downloads (several GB)
- Embedding generation is CPU-intensive
- Consider using GPU-enabled Ollama for better performance
- Large PDFs may require significant processing time