# RAG (Retrieval Augmented Generation) System

A Python-based RAG system that combines document retrieval with language model generation.

## Project Structure

- `rag.py` - Core RAG implementation
- `rag_ui.py` - User interface for the RAG system
- `model.txt` - Model configuration/weights
- `chunks.pkl` - Preprocessed document chunks
- `faiss.index` - FAISS vector index for document retrieval
- `requirements.txt` / `req.txt` - Project dependencies

## Setup

1. Create a virtual environment:
```sh
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

2. Install dependencies:
```sh
pip install -r requirements.txt
```

## Usage

1. Run the UI version:
```sh
python rag_ui.py
```

2. Use the RAG system programmatically:
```sh
python rag.py
```

## Requirements

- Python 3.8+
- See `requirements.txt` for full dependencies