from pathlib import Path
from langchain_community.vectorstores import FAISS

BASE_PATH = Path("vectorstores")


def get_vectorstore_path(user_id, doc_name):
    return BASE_PATH / f"user_{user_id}" / doc_name


def save_vector_store(vector_store, user_id, doc_name):
    path = get_vectorstore_path(user_id, doc_name)
    path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(path))


def load_vector_store(user_id, doc_name, embeddings):
    path = get_vectorstore_path(user_id, doc_name)
    if path.exists():
        return FAISS.load_local(
            str(path),
            embeddings,
            allow_dangerous_deserialization=True
        )
    return None
