import streamlit as st
import fitz
import numpy as np
import faiss
import pickle
import os
import requests

# Settings
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3"
OLLAMA_URL = "http://localhost:11434/api"
INDEX_FILE = "faiss.index"
CHUNKS_FILE = "chunks.pkl"

st.set_page_config(page_title="PDF RAG with Faiss & Ollama")

def load_pdf_chunks(file, chunk_size=300, overlap=50):
    pdf = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in pdf:
        text += page.get_text()
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def get_embedding(text):
    resp = requests.post(
        f"{OLLAMA_URL}/embeddings", 
        json={"model": EMBED_MODEL, "prompt": text}
    )
    resp.raise_for_status()
    return np.array(resp.json()["embedding"], dtype=np.float32)

def build_faiss_index(chunks):
    embeddings = np.vstack([get_embedding(chunk) for chunk in chunks])
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings

def save_cache(index, chunks):
    faiss.write_index(index, INDEX_FILE)
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)

def load_cache():
    index = faiss.read_index(INDEX_FILE)
    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def get_top_chunks(query, chunks, index, k=3):
    emb = get_embedding(query).reshape(1, -1)
    D, I = index.search(emb, k)
    return [chunks[i] for i in I[0]]

def ask_ollama(prompt: str) -> str:
    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False
        },
        timeout=60
    )
    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        print("HTTP error:", e)
        print("Response text:", resp.text)
        raise
    return resp.json().get("response", "")

def clear_cache():
    if os.path.exists(INDEX_FILE):
        os.remove(INDEX_FILE)
    if os.path.exists(CHUNKS_FILE):
        os.remove(CHUNKS_FILE)

# — UI —
st.title("PDF RAG with Ollama & Faiss")
st.markdown("Upload a PDF and ask questions from its context using Ollama & Faiss.")

with st.sidebar:
    st.header("Upload & Control")
    uploaded_pdf = st.file_uploader("Upload PDF", type="pdf")
    if st.button("Clear FAISS Index (remove all)"):
        clear_cache()
        st.success("Index/Caches cleared.")

# Session state keeps index/chunks for duration of app
if 'index' not in st.session_state or 'chunks' not in st.session_state:
    if os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE):
        st.session_state['index'], st.session_state['chunks'] = load_cache()
        st.info("Loaded cached index.")
    else:
        st.session_state['index'] = None
        st.session_state['chunks'] = None

if uploaded_pdf:
    st.write("Processing PDF, please wait...")
    chunks = load_pdf_chunks(uploaded_pdf)
    index, _ = build_faiss_index(chunks)
    save_cache(index, chunks)
    st.session_state['index'] = index
    st.session_state['chunks'] = chunks
    st.success(f"Indexed {len(chunks)} chunks from PDF.")

if st.session_state['index'] is not None:
    st.subheader("Ask a question")
    question = st.text_input("Enter your query")
    if question:
        with st.spinner("Retrieving answer..."):
            top_chunks = get_top_chunks(question, st.session_state['chunks'], st.session_state['index'])
            context = "\n\n".join(top_chunks)
            prompt = f"Answer the following question using this context:\n{context}\n\nQuestion: {question}\nAnswer:"
            answer = ask_ollama(prompt)
            st.markdown(f"**Answer:**\n{answer}")
            with st.expander("Show source text/chunks"):
                for i, c in enumerate(top_chunks):
                    st.markdown(f"**Chunk {i+1}**\n\n{c}")

else:
    st.info("Please upload a PDF to start.")

