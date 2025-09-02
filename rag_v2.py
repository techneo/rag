import streamlit as st
import fitz
import numpy as np
import faiss
import requests
import uuid

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3"
OLLAMA_URL = "http://localhost:11434/api"

def unique_id():
    return str(uuid.uuid4())[:8]

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
    return index

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

st.set_page_config(page_title="RAG PDF Chatbot (Session Only)")
st.title("PDF RAG: Multi-PDF, History, Session Only (No Disk)")

# Session state format
if 'pdfs' not in st.session_state:
    # {pdf_id: {'name':..., 'chunks':..., 'index':...}}
    st.session_state['pdfs'] = {}
if 'selected_pdf' not in st.session_state:
    st.session_state['selected_pdf'] = None
if 'history' not in st.session_state:
    # {pdf_id: [ (question, answer, [chunks]) ]}
    st.session_state['history'] = {}
if 'uploaded_filenames' not in st.session_state:
    st.session_state['uploaded_filenames'] = set()


with st.sidebar:
    st.header("Upload & Manage PDFs")
    uploaded_pdf = st.file_uploader("Upload multiple PDFs", type="pdf", accept_multiple_files=True)
    if uploaded_pdf:
        for file in uploaded_pdf:
            if file.name in st.session_state['uploaded_filenames']:
                continue  # Skip re-upload
            pdf_id = unique_id()
            chunks = load_pdf_chunks(file)
            index = build_faiss_index(chunks)
            st.session_state['pdfs'][pdf_id] = {"name": file.name, "chunks": chunks, "index": index}
            st.session_state['history'][pdf_id] = []
            st.session_state['selected_pdf'] = pdf_id
            st.session_state['uploaded_filenames'].add(file.name)
            st.success(f"Loaded: {file.name}")
    # List PDFs loaded in session
    st.subheader("Your PDFs this session")
    if st.session_state['pdfs']:
        for pdf_id, meta in st.session_state['pdfs'].items():
            if st.button(f"Select: {meta['name']}", key="sel_"+pdf_id):
                st.session_state['selected_pdf'] = pdf_id
            st.write(f"{meta['name']}")
            if st.button(f"Remove", key="del_"+pdf_id):
                st.session_state['pdfs'].pop(pdf_id)
                st.session_state['history'].pop(pdf_id)
                if st.session_state['selected_pdf'] == pdf_id:
                    # Select another if any left, else None
                    st.session_state['selected_pdf'] = next(iter(st.session_state['pdfs']), None)
    else:
        st.info("No PDFs uploaded this session.")

if st.session_state['selected_pdf']:
    pdf_id = st.session_state['selected_pdf']
    pdf_name = st.session_state['pdfs'][pdf_id]['name']
    chunks = st.session_state['pdfs'][pdf_id]['chunks']
    index = st.session_state['pdfs'][pdf_id]['index']
    st.header(f"Chat with: {pdf_name}")

    question = st.text_input("Ask a question (session-only, no persistence)")
    if question:
        top_chunks = get_top_chunks(question, chunks, index)
        context = "\n\n".join(top_chunks)
        prompt = f"Answer using this context:\n{context}\n\nQuestion: {question}\nAnswer:"
        answer = ask_ollama(prompt)
        st.session_state['history'][pdf_id].append((question, answer, top_chunks))
        st.markdown(f"**Answer:**\n{answer}")
        with st.expander("Show Source Chunks"):
            for i, c in enumerate(top_chunks):
                st.markdown(f"**Chunk {i+1}**\n\n{c}")

    st.subheader("History (for this PDF)")
    for idx, (q, a, tchs) in enumerate(reversed(st.session_state['history'].get(pdf_id, []))):
        with st.expander(f"Q{len(st.session_state['history'][pdf_id])-idx}: {q}"):
            st.markdown(f"**A:** {a}")
            for i, c in enumerate(tchs):
                st.markdown(f"**Source {i+1}**: {c}")

else:
    st.info("Please upload and select a PDF.")

