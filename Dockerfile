FROM python:3.13-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies for RAG and PDF processing
RUN pip install --no-cache-dir \
    langchain \
    langchain-community \
    langchain-ollama \
    chromadb \
    faiss-cpu \
    pypdf2 \
    pymupdf \
    pdfplumber \
    sentence-transformers \
    streamlit \
    gradio \
    python-dotenv \
    tiktoken \
    numpy \
    pandas

# Create directories for data and models
RUN mkdir -p /app/data /app/models /app/vectordb

# Expose ports for web interface
EXPOSE 8501 7860

# Copy application files
COPY . /app/
