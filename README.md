# DocuMind – Multi‑Document AI Knowledge Assistant

DocuMind is a Retrieval‑Augmented Generation (RAG) system that allows users to ask questions across single or multiple PDF documents with source attribution and confidence scoring.

## 🚀 Features
- Chat with single or multiple PDFs
- Semantic search using FAISS + embeddings
- MMR retrieval with local reranking
- Multi‑document answers with document‑level sources
- Smart fallback to general knowledge
- Confidence scoring for answer reliability
- Streamlit‑based interactive UI

## 🛠 Tech Stack
- Python
- Streamlit
- FAISS
- HuggingFace Embeddings
- Gemini API
- SQLite / Supabase (optional)

## ⚙️ Setup Instructions

```bash
git clone https://github.com/your-username/documind-ai
cd documind-ai
pip install -r requirements.txt
```

## Create a .env file:
- GOOGLE_API_KEY=your_api_key_here

## Run the app:
- streamlit run app.py

