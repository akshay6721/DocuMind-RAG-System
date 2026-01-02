import bcrypt
from db import get_db

# ---------------- PASSWORD UTILS ----------------

def hash_password(password: str) -> bytes:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt())

def verify_password(password: str, password_hash: bytes) -> bool:
    return bcrypt.checkpw(password.encode(), password_hash)

# ---------------- USER OPERATIONS ----------------

def create_user(email: str, password: str) -> bool:
    try:
        conn = get_db()
        cur = conn.cursor()

        password_hash = hash_password(password)

        cur.execute(
            "INSERT INTO users (email, password_hash) VALUES (?, ?)",
            (email, password_hash)
        )

        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()

def authenticate_user(email: str, password: str):
    conn = get_db()
    cur = conn.cursor()

    cur.execute(
        "SELECT * FROM users WHERE email = ?",
        (email,)
    )

    user = cur.fetchone()
    conn.close()

    if user and verify_password(password, user["password_hash"]):
        return dict(user)

    return None
