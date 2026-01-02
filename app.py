# ===============================
# IMPORTS
# ===============================
import streamlit as st
import os
import re
import torch
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import util

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

from db import (
    init_db,
    save_document,
    save_message,
    load_chat_history,
    get_user_documents
)

from auth import create_user, authenticate_user
from vector_store_utils import load_vector_store, save_vector_store

MAX_DOC_FAILURES = 2

# ===============================
# INITIALIZATION
# ===============================
init_db()
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    st.error("❌ Gemini API key not found in .env")
    st.stop()

genai.configure(api_key=API_KEY)

st.set_page_config(page_title="DocuMind AI", layout="wide")
st.title("🤖 DocuMind AI – PDF Chatbot")

# ===============================
# SESSION STATE (INITIALIZE ONCE)
# ===============================
if "user" not in st.session_state:
    st.session_state.user = None

if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {}

if "vector_stores" not in st.session_state:
    st.session_state.vector_stores = {}

if "current_doc" not in st.session_state:
    st.session_state.current_doc = None

if "current_topic" not in st.session_state:
    st.session_state.current_topic = ""

if "hydrated" not in st.session_state:
    st.session_state.hydrated = False

if "pending_fallback" not in st.session_state:
    st.session_state.pending_fallback = None

if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

if "doc_failure_count" not in st.session_state:
    st.session_state.doc_failure_count = 0



# ===============================
# PDF + VECTOR STORE HELPERS
# ===============================
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def create_vector_store(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.split_text(text)
    embeddings = st.session_state.embedding_model

    metadatas = [
        {"chunk_id": i}
        for i in range(len(chunks))
    ]

    return FAISS.from_texts(
        chunks,
        embedding=embeddings,
        metadatas=metadatas
    )


# ===============================
# QUERY + RERANKING HELPERS
# ===============================
def rewrite_query_locally(question: str) -> str:
    q = question.strip()
    vague_phrases = [
        "tell me more",
        "explain more",
        "what about this",
        "that project",
        "it"
    ]
    if any(p in q.lower() for p in vague_phrases) and st.session_state.current_topic:
        return f"{question} (Context: {st.session_state.current_topic})"
    return q


def rerank_chunks_locally(chunks, question, embedding_model, top_n=3):
    """
    Returns list of (chunk, score)
    """
    if not chunks:
        return []

    q_emb = embedding_model.embed_query(question)
    c_embs = embedding_model.embed_documents(chunks)

    scores = util.cos_sim(
        torch.tensor([q_emb]),
        torch.tensor(c_embs)
    )[0]

    ranked = sorted(
        zip(chunks, scores.tolist()),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked[:top_n]


# ===============================
# INTENT + ANSWER HELPERS
# ===============================
def detect_question_intent(question: str) -> str:
    q = question.lower()
    if any(k in q for k in ["summarize", "summary", "overview"]):
        return "summary"
    if any(k in q for k in ["list", "what are", "which"]):
        return "list"
    if any(k in q for k in ["explain", "describe", "how", "why"]):
        return "explain"
    if any(k in q for k in ["compare", "difference", "vs"]):
        return "compare"
    return "general"


def is_multi_document_query(question: str) -> bool:
    q = question.lower()

    keywords = [
        "all documents",
        "all pdfs",
        "across documents",
        "across pdfs",
        "which document",
        "compare documents",
        "compare pdfs",
        "multiple documents",
        "all files"
    ]

    return any(k in q for k in keywords)



def confidence_label(score: float) -> str:
    if score >= 0.75:
        return "High"
    elif score >= 0.5:
        return "Medium"
    else:
        return "Low"


def retrieve_multi_doc_chunks(user_id, question, top_k_per_doc=3):
    """
    Retrieve top chunks from ALL documents for a user.
    Returns list of (chunk, score, document_name)
    """
    all_ranked = []

    for doc_name, vector_store in st.session_state.vector_stores[user_id].items():
        docs = vector_store.max_marginal_relevance_search(
            question,
            k=top_k_per_doc * 2,
            fetch_k=20,
            lambda_mult=0.7
        )

        chunks = [d.page_content for d in docs]

        ranked = rerank_chunks_locally(
            chunks,
            question,
            st.session_state.embedding_model,
            top_n=top_k_per_doc
        )

        for chunk, score in ranked:
            all_ranked.append((chunk, score, doc_name))

    # Global rerank across documents
    all_ranked.sort(key=lambda x: x[1], reverse=True)

    return all_ranked



def get_smart_answer(context, question):
    model = genai.GenerativeModel("gemini-2.5-flash")
    intent = detect_question_intent(question)

    prompt = f"""
You are DocuMind AI.

Rules:
- Use ONLY the document context
- Never hallucinate
- If answer is missing, say so clearly

Context:
{context}

Question:
{question}

Answer style:
{intent}

Answer:
"""

    try:
        response = model.generate_content(prompt)
        return response.text
    except ResourceExhausted:
        return "⚠️ API quota exceeded. Please try again later."


def get_general_chat_answer(question: str):
    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = f"""
You are a friendly and intelligent assistant.

User question:
{question}

Answer clearly and concisely:
"""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception:
        return "⚠️ I’m temporarily unavailable. Please try again."


# ======================================================
# AUTHENTICATION (BLOCK APP UNTIL LOGIN)
# ======================================================
if st.session_state.user is None:
    st.sidebar.subheader("🔐 Authentication")

    email = st.sidebar.text_input("Email", key="auth_email")
    password = st.sidebar.text_input(
        "Password", type="password", key="auth_password"
    )

    col1, col2 = st.sidebar.columns(2)

    with col1:
        if st.button("Login"):
            user = authenticate_user(email, password)
            if user:
                st.session_state.user = user
                st.session_state.current_doc = None
                st.session_state.hydrated = False
                st.rerun()
            else:
                st.sidebar.error("Invalid email or password")

    with col2:
        if st.button("Sign up"):
            if create_user(email, password):
                st.sidebar.success("Account created. Please log in.")
            else:
                st.sidebar.error("User already exists")

    st.stop()

# ======================================================
# SESSION HYDRATION (RUNS ONCE PER LOGIN)
# ======================================================
if st.session_state.user and not st.session_state.hydrated:
    user_id = st.session_state.user["id"]

    st.session_state.chat_histories.setdefault(user_id, {})
    st.session_state.vector_stores.setdefault(user_id, {})

    # Load general chat
    general_history = load_chat_history(user_id, document_name=None)
    if general_history:
        st.session_state.chat_histories[user_id]["GENERAL_CHAT"] = general_history

    # Load document chats + vector stores
    docs = get_user_documents(user_id)
    for doc_name in docs:
        st.session_state.chat_histories[user_id][doc_name] = load_chat_history(
            user_id, document_name=doc_name
        )

        if doc_name not in st.session_state.vector_stores[user_id]:
            vs = load_vector_store(
                user_id, doc_name, st.session_state.embedding_model
            )
            if vs:
                st.session_state.vector_stores[user_id][doc_name] = vs

    if docs:
        st.session_state.current_doc = docs[-1]

    st.session_state.hydrated = True

# ======================================================
# SIDEBAR — PDF UPLOAD + SELECT
# ======================================================
st.sidebar.header("📄 Upload PDFs")

uploaded_files = st.sidebar.file_uploader(
    "Upload one or more PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

user_id = st.session_state.user["id"]

if uploaded_files:
    with st.spinner("Processing PDFs..."):
        for file in uploaded_files:
            if file.name in st.session_state.vector_stores[user_id]:
                continue

            vs = load_vector_store(
                user_id, file.name, st.session_state.embedding_model
            )

            if vs is None:
                text = extract_text_from_pdf(file)
                vs = create_vector_store(text)
                save_vector_store(vs, user_id, file.name)

            st.session_state.vector_stores[user_id][file.name] = vs
            st.session_state.chat_histories[user_id].setdefault(file.name, [])
            save_document(user_id, file.name)

    if st.session_state.current_doc is None:
        st.session_state.current_doc = next(
            iter(st.session_state.vector_stores[user_id])
        )

    st.sidebar.success("PDFs processed successfully!")

# Document selector
docs_for_user = list(st.session_state.vector_stores[user_id].keys())
if docs_for_user:
    st.sidebar.subheader("📂 Select Document")
    st.session_state.current_doc = st.sidebar.selectbox(
        "Choose a document",
        options=docs_for_user,
        index=docs_for_user.index(st.session_state.current_doc)
        if st.session_state.current_doc in docs_for_user else 0
    )

# Logout
st.sidebar.divider()
if st.sidebar.button("🚪 Logout"):
    st.session_state.user = None
    st.session_state.hydrated = False
    st.rerun()

# ======================================================
# CHAT UI
# ======================================================
st.subheader("💬 Chat with DocuMind AI")

if st.session_state.current_doc:
    st.caption(f"📄 Asking about: {st.session_state.current_doc}")

with st.form(key="chat_form", clear_on_submit=True):
    user_question = st.text_input(
        "Ask anything — upload documents for deeper analysis"
    )
    submitted = st.form_submit_button("Send")

# ======================================================
# CHAT PROCESSING (RUNS ONLY ON SUBMIT)
# ======================================================
if submitted and user_question.strip():
    use_document_mode = st.session_state.current_doc is not None

    # ---------------- GENERAL CHAT ----------------
    if not use_document_mode:
        st.session_state.chat_histories[user_id].setdefault("GENERAL_CHAT", [])

        answer = get_general_chat_answer(user_question)

        st.session_state.chat_histories[user_id]["GENERAL_CHAT"].extend([
            ("You", user_question),
            ("DocuMind (General)", answer)
        ])

        save_message(user_id, None, "You", user_question)
        save_message(user_id, None, "DocuMind (General)", answer)

    # ---------------- DOCUMENT MODE (RAG) ----------------
    else:
        current_doc = st.session_state.current_doc
        vector_store = st.session_state.vector_stores[user_id][current_doc]
        history = st.session_state.chat_histories[user_id][current_doc]

        # 🔍 Detect intent EARLY
        intent = detect_question_intent(user_question)

        # 🔍 Multi‑document detection
        multi_doc_mode = (
            is_multi_document_query(user_question)
            or st.session_state.current_doc is None
        )

        if multi_doc_mode:
            st.markdown("🔍 **Searching across all documents**")

        # Clean query
        rewritten_query = rewrite_query_locally(user_question)

        # ==================================================
        # 🔎 RETRIEVAL
        # ==================================================
        if multi_doc_mode:
            ranked_multi = retrieve_multi_doc_chunks(
                user_id=user_id,
                question=user_question,
                top_k_per_doc=2
            )

            MIN_SCORE = 0.30
            good_chunks = [
                chunk for chunk, score, _ in ranked_multi if score >= MIN_SCORE
            ]

            used_docs = list({doc for _, _, doc in ranked_multi})

        else:
            docs = vector_store.max_marginal_relevance_search(
                rewritten_query, k=8, fetch_k=20, lambda_mult=0.7
            )

            candidate_chunks = [d.page_content for d in docs]

            ranked = rerank_chunks_locally(
                candidate_chunks,
                user_question,
                st.session_state.embedding_model
            )

            MIN_SCORE = 0.30
            good_chunks = [
                chunk for chunk, score in ranked if score >= MIN_SCORE
            ]

        # ==================================================
        # 🚫 DOCUMENT‑ONLY INTENTS (SUMMARY / OVERVIEW)
        # ==================================================
        DOCUMENT_ONLY_INTENTS = {"summary", "general"}

        if intent in DOCUMENT_ONLY_INTENTS:
            context = "\n\n".join(
                candidate_chunks[:5] if not multi_doc_mode else good_chunks[:5]
            )

            answer = get_smart_answer(context, user_question)

            history.extend([
                ("You", user_question),
                ("DocuMind (Document)", answer)
            ])

            save_message(user_id, current_doc, "You", user_question)
            save_message(user_id, current_doc, "DocuMind (Document)", answer)

            st.session_state.doc_failure_count = 0

        # ==================================================
        # SMART FALLBACK (FACTUAL MISS ONLY)
        # ==================================================
        elif not good_chunks and st.session_state.pending_fallback is None:

            st.session_state.doc_failure_count += 1

            # 🔁 AUTO‑SWITCH TO GENERAL AFTER N FAILURES
            if st.session_state.doc_failure_count >= MAX_DOC_FAILURES:
                auto_msg = (
                    "🔁 I couldn’t find answers to multiple questions in this document.\n\n"
                    "Switching to general knowledge to help you."
                )

                history.append(("DocuMind (System)", auto_msg))
                save_message(user_id, current_doc, "DocuMind (System)", auto_msg)

                answer = get_general_chat_answer(user_question)

                history.extend([
                    ("You", user_question),
                    ("DocuMind (General)", answer)
                ])

                save_message(user_id, current_doc, "You", user_question)
                save_message(user_id, current_doc, "DocuMind (General)", answer)

                st.session_state.doc_failure_count = 0

            # 🧠 NORMAL PERMISSION FLOW
            else:
                fallback_msg = (
                    "📄 I couldn’t find this information in the selected document.\n\n"
                    "Would you like me to answer using general knowledge?"
                )

                history.append(("DocuMind (Document)", fallback_msg))

                st.session_state.pending_fallback = {
                    "question": user_question,
                    "doc": current_doc
                }

                save_message(
                    user_id,
                    current_doc,
                    "DocuMind (Document)",
                    fallback_msg
                )

        # ==================================================
        # NORMAL DOCUMENT ANSWER (SOURCES + CONFIDENCE)
        # ==================================================
        elif good_chunks:
            context = "\n\n".join(good_chunks)
            answer = get_smart_answer(context, user_question)

            # ==========================
            # MULTI‑DOCUMENT FORMATTING
            # ==========================
            if multi_doc_mode:
                sources_text = "\n".join(
                    [f"• {doc}" for doc in used_docs]
                )

                used_scores = [
                    score for chunk, score, _ in ranked_multi if chunk in good_chunks
                ]

            # ==========================
            # SINGLE‑DOCUMENT FORMATTING
            # ==========================
            else:
                sources_text = "\n".join(
                    [f"• Source chunk #{i + 1}" for i in range(len(good_chunks))]
                )

                used_scores = [
                    score for chunk, score in ranked if chunk in good_chunks
                ]

            confidence_score = round(
                sum(used_scores) / len(used_scores), 2
            )
            confidence_text = confidence_label(confidence_score)

            final_answer = f"""{answer}

---
📌 **Sources used:**
{sources_text}

📊 **Confidence:** {confidence_text} ({confidence_score})
"""

            history.extend([
                ("You", user_question),
                ("DocuMind (Document)", final_answer)
            ])

            save_message(user_id, current_doc, "You", user_question)
            save_message(user_id, current_doc, "DocuMind (Document)", final_answer)

            # ✅ Reset failure counter
            st.session_state.doc_failure_count = 0


# ======================================================
# DISPLAY CHAT HISTORY
# ======================================================
history = (
    st.session_state.chat_histories[user_id].get(
        st.session_state.current_doc, []
    )
    if st.session_state.current_doc
    else st.session_state.chat_histories[user_id].get("GENERAL_CHAT", [])
)

if history:
    for role, msg in history:
        if role == "You":
            st.markdown(f"**🧑 You:** {msg}")
        else:
            st.markdown(f"**🤖 {role}:** {msg}")
else:
    st.info("Start chatting with DocuMind AI 🙂")

# ======================================================
# SMART FALLBACK PERMISSION UI
# ======================================================
if st.session_state.pending_fallback:
    st.markdown("### 🔍 Use General Knowledge?")
    c1, c2 = st.columns(2)

    with c1:
        if st.button("✅ Answer using Gemini"):
            q = st.session_state.pending_fallback["question"]
            doc = st.session_state.pending_fallback["doc"]

            answer = get_general_chat_answer(q)

            st.session_state.chat_histories[user_id][doc].append(
                ("DocuMind (General)", answer)
            )

            save_message(user_id, doc, "DocuMind (General)", answer)
            st.session_state.pending_fallback = None
            st.rerun()

    with c2:
        if st.button("❌ Stay document-only"):
            st.session_state.pending_fallback = None
            st.rerun()
