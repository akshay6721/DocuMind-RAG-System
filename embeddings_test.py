# embeddings_test.py
# ------------------------------------
# Local Embeddings + FAISS Test Script
# ------------------------------------

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# ---------------- CONFIG ----------------
PDF_PATH = "sample.pdf"
QUERY = "What projects are mentioned?"


# ---------------- LOAD PDF ----------------
reader = PdfReader(PDF_PATH)
text = ""

for page in reader.pages:
    page_text = page.extract_text()
    if page_text:
        text += page_text + "\n"

print("PDF loaded successfully")


# ---------------- CHUNKING ----------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200
)

chunks = text_splitter.split_text(text)
print(f"Total chunks created: {len(chunks)}")


# ---------------- LOCAL EMBEDDINGS ----------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embeddings model loaded")


# ---------------- VECTOR STORE ----------------
vector_store = FAISS.from_texts(chunks, embedding=embeddings)
print("FAISS vector store created")


# ---------------- SEMANTIC SEARCH ----------------
docs = vector_store.similarity_search(QUERY, k=8)

print("\nTop Relevant Chunks:\n")

for i, doc in enumerate(docs):
    print(f"--- Chunk {i+1} ---")
    print(doc.page_content[:500])
    print()
