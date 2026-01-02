import sqlite3
from pathlib import Path

DB_PATH = Path("documind.db")


# ---------------- DB CONNECTION ----------------
def get_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------- INIT DATABASE ----------------
def init_db():
    conn = get_db()
    cur = conn.cursor()

    # USERS
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # DOCUMENTS
    cur.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(user_id, name)
    )
    """)

    # CHAT MESSAGES
    cur.execute("""
    CREATE TABLE IF NOT EXISTS chat_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        document_name TEXT,
        role TEXT NOT NULL,
        message TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()


# ---------------- DOCUMENTS ----------------
def save_document(user_id, name):
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        INSERT OR IGNORE INTO documents (user_id, name)
        VALUES (?, ?)
    """, (user_id, name))

    conn.commit()
    conn.close()


def get_user_documents(user_id):
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        SELECT name FROM documents
        WHERE user_id = ?
        ORDER BY created_at
    """, (user_id,))

    docs = [row["name"] for row in cur.fetchall()]
    conn.close()
    return docs


# ---------------- CHAT MESSAGES ----------------
def save_message(user_id, document_name, role, message):
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO chat_messages
        (user_id, document_name, role, message)
        VALUES (?, ?, ?, ?)
    """, (user_id, document_name, role, message))

    conn.commit()
    conn.close()


def load_chat_history(user_id, document_name=None, limit=50):
    conn = get_db()
    cur = conn.cursor()

    if document_name:
        cur.execute("""
            SELECT role, message FROM chat_messages
            WHERE user_id = ? AND document_name = ?
            ORDER BY created_at
            LIMIT ?
        """, (user_id, document_name, limit))
    else:
        cur.execute("""
            SELECT role, message FROM chat_messages
            WHERE user_id = ? AND document_name IS NULL
            ORDER BY created_at
            LIMIT ?
        """, (user_id, limit))

    rows = cur.fetchall()
    conn.close()

    return [(row["role"], row["message"]) for row in rows]
