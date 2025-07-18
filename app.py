import os
import glob
import streamlit as st
from PyPDF2 import PdfReader
import requests
import chromadb
from tqdm import tqdm


OLLAMA_HOST = os.environ.get("OLLAMA_HOST")
EMBED_MODEL = "mxbai-embed-large:latest"  # For embeddings
CHAT_MODEL  = "gemma3n:e2b"
DATA_DIR    = "data"       # mounted to /app/data
VDB_DIR     = "vectordb"   # mounted to /app/vectordb


if OLLAMA_HOST:
    st.success(f"Ollama is configured at: {OLLAMA_HOST}")
else:
    st.error("OLLAMA_HOST environment variable is not set.")

st.write("Application is ready.")

# --- PDF Extraction ---
def extract_text_from_pdf(path):
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

# --- Chunking ---
def chunk_text(text, size=800, overlap=100):
    """
    Improved chunking that preserves paragraph structure and context
    """
    chunks = []
    
    # First, split by double newlines (paragraphs)
    paragraphs = text.split('\n\n')
    
    current_chunk = ""
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        # If adding this paragraph would exceed size, save current chunk
        if len(current_chunk + paragraph) > size and current_chunk:
            chunks.append(current_chunk.strip())
            
            # Start new chunk with overlap from previous paragraph if it's long enough
            if len(paragraph) > overlap:
                current_chunk = paragraph
            else:
                current_chunk = current_chunk[-overlap:] + "\n\n" + paragraph
        else:
            # Add paragraph to current chunk
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
    
    # Add the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Handle very long paragraphs that exceed chunk size
    final_chunks = []
    for chunk in chunks:
        if len(chunk) <= size:
            final_chunks.append(chunk)
        else:
            # Split long paragraphs by sentences, preserving context
            sentences = chunk.split('. ')
            temp_chunk = ""
            
            for sentence in sentences:
                if len(temp_chunk + sentence) > size and temp_chunk:
                    final_chunks.append(temp_chunk.strip())
                    temp_chunk = sentence + ". "
                else:
                    temp_chunk += sentence + ". "
            
            if temp_chunk.strip():
                final_chunks.append(temp_chunk.strip())
    
    return final_chunks

# --- Ollama Embeddings ---
def embed_texts(texts):
    embs = []
    for t in tqdm(texts, desc="Embedding"):
        resp = requests.post(
            f"{OLLAMA_HOST}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": t}
        )
        if resp.status_code != 200:
            st.error(f"Embedding API Error {resp.status_code}: {resp.text}")
            return []
        embs.append(resp.json()["embedding"])
    return embs

# --- ChromaDB Storage ---
def get_collection():
    client = chromadb.PersistentClient(path=VDB_DIR)
    return client.get_or_create_collection("pdf_rag")
    
def query_rag(q, col, k=3):
    # embed query
    resp = requests.post(
        f"{OLLAMA_HOST}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": q}
    )
    if resp.status_code != 200:
        st.error(f"Embedding API Error {resp.status_code}: {resp.text}")
        return "Error generating embeddings"
    
    q_emb = resp.json()["embedding"]
    # retrieve
    res = col.query(query_embeddings=[q_emb], n_results=k)

    # Enhanced context with source information
    docs = res["documents"][0]
    st.session_state["last_retrieved_docs"] = docs
    metas = res["metadatas"][0] if res["metadatas"] else []
    distances = res["distances"][0] if res["distances"] else []

    ctx_parts = []
    source_info = []
    for i, doc in enumerate(docs):
        source = metas[i].get("source", "Unknown") if i < len(metas) else "Unknown"
        distance = distances[i] if i < len(distances) else 0
        
        similarity = max(0, 1 - (distance / 2.0))
        
        ctx_parts.append(f"[Source: {source}]\n{doc}")
        source_info.append({"source": source, "similarity": similarity, "distance": distance})

    ctx = "\n\n".join(ctx_parts)

    # ask LLM (same as before)
    prompt = f"""You are a helpful assistant that answers questions based solely on the provided context from PDF documents.

                Instructions:
                - Answer the question using only the information from the context below
                - If the context doesn't contain enough information to answer the question, say "I don't have enough information in the provided documents to answer this question"
                - Be concise but comprehensive in your response
                - Cite specific details from the context when possible
                - Do not make up information that isn't in the context

                Context:
                {ctx}

                Question: {q}

                Answer:"""
    
    resp = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={"model": CHAT_MODEL, "prompt": prompt, "stream": False}
    )
    
    if resp.status_code != 200:
        st.error(f"Generation API Error {resp.status_code}: {resp.text}")
        return "Error generating response", []
    
    try:
        return resp.json()["response"], source_info
    except Exception as e:
        st.error(f"Error parsing response: {e}")
        return "Error parsing response", []


# --- Check Model Availability ---
def check_models_available():
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags")
        if resp.status_code == 200:
            models = [model["name"] for model in resp.json().get("models", [])]
            embed_ready = EMBED_MODEL in models
            chat_ready = CHAT_MODEL in models
            return embed_ready, chat_ready
        return False, False
    except:
        return False, False

def get_indexed_sources():
    """Get list of sources (PDF filenames) that are already indexed"""
    try:
        col = get_collection()
        if col.count() == 0:
            return set()
        
        # Get all metadata to see which sources are indexed
        all_data = col.get()
        if all_data and all_data["metadatas"]:
            sources = set(meta.get("source", "") for meta in all_data["metadatas"])
            return sources
        return set()
    except:
        return set()

def index_pdfs():
    docs, metas = [], []
    
    # Check which PDFs we're about to process
    pdf_files = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    pdf_names = [os.path.basename(pdf) for pdf in pdf_files]
    
    # Check which PDFs are already indexed
    indexed_sources = get_indexed_sources()
    new_pdfs = [pdf for pdf in pdf_files if os.path.basename(pdf) not in indexed_sources]
    already_indexed = [pdf for pdf in pdf_files if os.path.basename(pdf) in indexed_sources]
    
    st.info(f"📖 Found {len(pdf_files)} PDF files: {', '.join(pdf_names)}")
    
    if already_indexed:
        indexed_names = [os.path.basename(pdf) for pdf in already_indexed]
        st.info(f"✅ Already indexed: {', '.join(indexed_names)}")
    
    if not new_pdfs:
        st.success("🎉 All PDFs are already indexed! Ready to chat.")
        # Load existing collection if not already loaded
        if "collection" not in st.session_state:
            st.session_state["collection"] = get_collection()
        return
    
    new_pdf_names = [os.path.basename(pdf) for pdf in new_pdfs]
    st.info(f"🔄 Processing new PDFs: {', '.join(new_pdf_names)}")
    
    for pdf in new_pdfs:
        text = extract_text_from_pdf(pdf)
        chunks = chunk_text(text)
        docs.extend(chunks)
        metas.extend([{"source": os.path.basename(pdf)}] * len(chunks))
    
    if not docs:
        st.warning("No text chunks found in new PDFs")
        return
        
    st.info(f"📄 Processing {len(docs)} new text chunks...")
    embs = embed_texts(docs)
    
    if not embs:
        st.error("Failed to generate embeddings")
        return
        
    col = get_collection()
    
    # Generate unique IDs that don't conflict with existing ones
    existing_count = col.count()
    new_ids = [f"c{existing_count + i}" for i in range(len(docs))]
    
    col.add(documents=docs, embeddings=embs, metadatas=metas, ids=new_ids)
    st.session_state["collection"] = col
    
    total_count = col.count()
    st.success(f"✅ Successfully indexed {len(docs)} new chunks from {len(new_pdfs)} PDFs!")
    st.success(f"📊 Total chunks in database: {total_count}")

def check_existing_index():
    """Check if PDFs are already indexed"""
    try:
        col = get_collection()
        count = col.count()
        if count > 0:
            indexed_sources = get_indexed_sources()
            return col, indexed_sources
        return None, set()
    except:
        return None, set()


# --- Streamlit App ---

st.set_page_config(page_title="Local PDF RAG Chatbot")
st.title("📄🤖 Local PDF RAG Chatbot")

embed_ready, chat_ready = check_models_available()

if embed_ready and chat_ready:
    st.success(f"✅ Both models ready: {EMBED_MODEL} & {CHAT_MODEL}")
elif embed_ready:
    st.warning(f"⏳ Embedding model ready, waiting for chat model {CHAT_MODEL}")
elif chat_ready:
    st.warning(f"⏳ Chat model ready, waiting for embedding model {EMBED_MODEL}")
else:
    st.error("⏳ Waiting for both models to download...")

st.write("This is a simple RAG application running with Streamlit.")

# Check for existing index on startup
if "collection" not in st.session_state and embed_ready:
    existing_collection, indexed_sources = check_existing_index()
    if existing_collection:
        st.session_state["collection"] = existing_collection
        count = existing_collection.count()
        if indexed_sources:
            source_list = ', '.join(indexed_sources)
            st.info(f"📚 Found existing index with {count} chunks from: {source_list}")
        else:
            st.info(f"📚 Found existing index with {count} chunks. Ready to chat!")

if embed_ready:
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Index all PDFs"):
            index_pdfs()
    
    with col2:
        if "collection" in st.session_state:
            if st.button("🗑️ Clear Index"):
                col = get_collection()
                # Get all IDs first, then delete them
                all_data = col.get()
                if all_data and all_data["ids"]:
                    col.delete(ids=all_data["ids"])
                    st.success("Index cleared!")
                else:
                    st.info("Index was already empty")
                
                if "collection" in st.session_state:
                    del st.session_state["collection"]
                st.rerun()
else:
    st.info("Waiting for embedding model to be ready...")

# Show current status
if "collection" in st.session_state:
    col = st.session_state["collection"]
    count = col.count()
    st.success(f"📊 Vector database loaded: {count} chunks indexed")
    
    if chat_ready:
        query = st.text_input("Ask a question about your PDFs")
        if st.button("❓ Ask") and query:
            with st.spinner("⏳ Thinking..."):
                answer, source_info = query_rag(query, st.session_state["collection"])
            
            st.markdown(f"**Answer:** {answer}")
            
            if source_info:
                st.markdown("**📚 Sources Used:**")
                for info in source_info:
                    distance = info.get("distance", 0)
                    similarity = info.get("similarity", 0)
                    if distance > 100:
                        quality = "❌ Very Poor Match"
                    elif distance > 50:
                        quality = "⚠️ Poor Match"
                    elif distance > 10:
                        quality = "🟡 Moderate Match"
                    else:
                        quality = "✅ Good Match"
                    
                    st.markdown(f"- {info['source']} (distance: {distance:.1f}) {quality} {similarity}")
    else:
        st.info("Waiting for chat model to be ready...")
else:
    st.info("Click **Index all PDFs** to get started.")

# Debug info
with st.expander("🔍 Debug Info"):
    pdf_count = len(glob.glob(os.path.join(DATA_DIR, "*.pdf")))
    st.write(f"📁 PDFs in data folder: {pdf_count}")
    
    
    if st.button("Debug Last Query"):
        if "last_retrieved_docs" in st.session_state:
            for i, doc in enumerate(st.session_state["last_retrieved_docs"]):
                st.write(f"**Chunk {i+1}:**")
                st.write(doc[:200] + "..." if len(doc) > 200 else doc)
                st.write("---")
        else:
            st.warning("No last query found in session state")
    # Test embedding
    if st.button("Test Embedding"):
        test_resp = requests.post(
            f"{OLLAMA_HOST}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": "test query"}
        )
        if test_resp.status_code == 200:
            embedding = test_resp.json()["embedding"]
            st.write(f"Embedding dimension: {len(embedding)}")
            st.write(f"Sample values: {embedding[:5]}")
        else:
            st.error(f"Embedding test failed: {test_resp.text}")    
    
    if "collection" in st.session_state:
        col = st.session_state["collection"]
        count = col.count()
        st.write(f"💾 Chunks in vector database: {count}")
        
        # Show indexed sources
        indexed_sources = get_indexed_sources()
        if indexed_sources:
            st.write(f"📚 Indexed PDFs: {', '.join(indexed_sources)}")
            
            # Show which PDFs are new
            pdf_files = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
            pdf_names = set(os.path.basename(pdf) for pdf in pdf_files)
            new_pdfs = pdf_names - indexed_sources
            if new_pdfs:
                st.write(f"🆕 New PDFs to index: {', '.join(new_pdfs)}")
            else:
                st.write("✅ All PDFs are indexed")
        else:
            st.write("📚 No PDFs indexed yet")
    else:
        st.write("💾 No vector database loaded")

st.sidebar.markdown(f"**Ollama host:** {OLLAMA_HOST}")
st.sidebar.markdown(f"**Embedding model:** {EMBED_MODEL}")
st.sidebar.markdown(f"**Chat model:** {CHAT_MODEL}")