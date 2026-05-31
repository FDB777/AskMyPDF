# AskMyPDF 📄

An AI-powered PDF assistant that lets you upload multiple PDFs and ask questions in natural language. Built with a full RAG (Retrieval-Augmented Generation) pipeline.

## Features
- 📂 Upload multiple PDFs at once
- 🔍 Vector search using FAISS
- 🔄 Query rewriting for better retrieval
- 📊 Keyword reranking on top of vector search
- 💬 Full conversation memory across turns
- 📚 Sources with page numbers for every answer

## How it works
1. PDFs are parsed and split into overlapping sentence chunks
2. Chunks are embedded using `all-MiniLM-L6-v2` and stored in a FAISS index
3. User query is rewritten by GPT-4o-mini for better search
4. Top chunks are retrieved by vector similarity + keyword reranking
5. GPT-4o-mini generates a grounded answer with page citations

## Tech Stack
- Python, Streamlit
- FAISS, Sentence Transformers
- PyPDF, NLTK
- GitHub Models API (GPT-4o-mini)

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/FDB777/AskMyPDF.git
cd AskMyPDF
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your API key
Create a `.streamlit/secrets.toml` file:
```toml
GITHUB_TOKEN = "your_github_token_here"
```
> ⚠️ The API key is NOT included in this repo for security reasons. Get your free GitHub Models token at [github.com/marketplace/models](https://github.com/marketplace/models)

### 4. Run the app
```bash
streamlit run app.py
```

## Requirements
```
streamlit
sentence-transformers
pypdf
faiss-cpu
openai
nltk
numpy
```
