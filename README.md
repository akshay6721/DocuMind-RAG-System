# DocuMind – Multi‑Document AI Knowledge Assistant

DocuMind is a Retrieval‑Augmented Generation (RAG) system that allows users to ask questions across single or multiple PDF documents with source attribution and confidence scoring.

## 🚀 Features
- 📚 Single & Multi‑Document Question Answering
- 🔎 Semantic Search using FAISS + HuggingFace embeddings
- 🎯 MMR Retrieval with local cosine‑similarity re‑ranking
- 📌 Source Attribution (document‑level & chunk‑level)
- 📊 Confidence Scoring for answer reliability
- 🧠 Smart Fallback System
-    Switches to general knowledge only with user permission
-    Auto‑switch after repeated document failures
- 🧱 Graceful Degradation
-    Retrieval works even when LLM API quota is exceeded
- 🖥️ Interactive Streamlit UI
-    PDF upload & selection
-    Persistent chat history
-    Dynamic query routing

## 🛠 Tech Stack
- Language: Python
- Frontend: Streamlit
- Embeddings: HuggingFace (Sentence Transformers)
- Vector Store: FAISS
- LLM: Google Gemini API
- PDF Parsing: PyPDF
- Similarity Scoring: Cosine Similarity (Local)
- Database: SQLite (chat history & documents)

## 🧠 System Architecture

```bash
User Query
   ↓
Intent Detection (Summary / Factual / Multi‑Doc)
   ↓
Vector Retrieval (FAISS + MMR)
   ↓
Local Re‑Ranking (Cosine Similarity)
   ↓
Context Assembly
   ↓
LLM Generation (Gemini)
   ↓
Sources + Confidence Score
```

## 📂 Project Structure

```bash
DocuMind/
├── app.py
├── auth.py
├── db.py
├── vector_store_utils.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## ⚙️ Setup & installation

### 1️⃣ Clone the Repository : 
```bash
git clone https://github.com/your-username/documind-ai.git
cd documind-ai
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Environment Variables

- Create a .env file:
```bash
GOOGLE_API_KEY=your_gemini_api_key
```

### 5️⃣ Run the App

```bash
streamlit run app.py
```

## 🧪 Example Use Cases

- “What is this document about?”
- “Which document mentions Artificial Intelligence?”
- “Summarize all uploaded PDFs”
- “Compare topics across multiple documents”

## 🔒 Privacy & Security

- API keys are never committed to the repository
- .gitignore excludes .env, vector stores, and local DB files
- All document embeddings are stored locally

## 📈 Why This Project Matters

### This project demonstrates:
- Real‑world RAG system design
- Strong understanding of LLM limitations & fallback strategies
- Production‑thinking beyond simple chatbot demos
- Clean separation of retrieval vs generation


## 🧑‍💻 Author

Akshay Umbarge
Computer Engineering | AI & Backend Enthusiast
📫 Email: akshayumbargе6721@gmail.com
🔗 LinkedIn: https://www.linkedin.com/in/akshay-umbarge-5b185a1bb/