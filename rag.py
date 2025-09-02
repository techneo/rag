import fitz  # PyMuPDF
import numpy as np
import faiss
import pickle
import os
import requests

# SETTINGS
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3"
OLLAMA_URL = "http://localhost:11434/api"

def load_pdf_chunks(pdf_path, chunk_size=300, overlap=50):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    # Simple whitespace splitting
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
    return np.array(resp.json()["embedding"], dtype='float32')

def build_faiss_index(chunks):
    embeddings = np.vstack([get_embedding(chunk) for chunk in chunks])
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings

def save_cache(index, chunks, index_file="faiss.index", chunks_file="chunks.pkl"):
    faiss.write_index(index, index_file)
    with open(chunks_file, "wb") as f:
        pickle.dump(chunks, f)

def load_cache(index_file="faiss.index", chunks_file="chunks.pkl"):
    index = faiss.read_index(index_file)
    with open(chunks_file, "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def get_top_chunks(query, chunks, index, k=3):
    emb = get_embedding(query).reshape(1, -1)
    D, I = index.search(emb, k)
    return [chunks[i] for i in I[0]]

def ask_ollama(prompt):
    resp = requests.post(
        f"{OLLAMA_URL}/generate",
        json={"model": LLM_MODEL, "prompt": prompt}
    )
    resp.raise_for_status()
    return resp.json()['response']

def main(pdf_path):
    if os.path.exists("faiss.index") and os.path.exists("chunks.pkl"):
        index, chunks = load_cache()
        print("[+] Loaded cached index and chunks.")
    else:
        print("[+] Loading and chunking PDF...")
        chunks = load_pdf_chunks(pdf_path)
        print(f"[+] {len(chunks)} chunks loaded")
        print("[+] Generating embeddings and building index...")
        index, _ = build_faiss_index(chunks)
        save_cache(index, chunks)
        print("[+] Saved FAISS index and chunks for reuse")
    
    while True:
        query = input("Your question (or 'exit'): ")
        if query.lower() == "exit":
            break
        top_chunks = get_top_chunks(query, chunks, index)
        context = "\n\n".join(top_chunks)
        prompt = f"Answer the following question using this context:\n{context}\n\nQuestion: {query}\nAnswer:"
        answer = ask_ollama(prompt)
        print("Answer:", answer)
        
if __name__ == "__main__":
    main("your-file.pdf")  # Replace with your PDF
