"""Conversation storage for DarkGPT.

Two tables, both in the existing `chat_history.db`:

    conversations  one row per chat thread (title, model, timestamps)
    messages       one row per turn, ordered by id

The upstream repo's `chat_history` table is left exactly as it was found: it is
read once, copied into the new shape, and then never written to again. That keeps
whatever conversation the previous version recorded without destroying anything.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "chat_history.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    title      TEXT NOT NULL DEFAULT 'New chat',
    model      TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    attachments     TEXT,
    created_at      TEXT NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation
    ON messages(conversation_id, id);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    """Create the schema and import the legacy table exactly once."""
    conn = connect()
    try:
        conn.executescript(_SCHEMA)
        conn.commit()
        _import_legacy(conn)
    finally:
        conn.close()


def _import_legacy(conn: sqlite3.Connection) -> None:
    done = conn.execute("SELECT value FROM meta WHERE key='legacy_imported'").fetchone()
    if done:
        return

    exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='chat_history'"
    ).fetchone()
    if not exists:
        conn.execute("INSERT INTO meta(key, value) VALUES('legacy_imported', '1')")
        conn.commit()
        return

    rows = conn.execute(
        "SELECT conversation_id, role, content FROM chat_history ORDER BY rowid"
    ).fetchall()
    groups: Dict[int, List[sqlite3.Row]] = {}
    for row in rows:
        groups.setdefault(row["conversation_id"], []).append(row)

    for legacy_id, items in groups.items():
        title = "Imported chat"
        for item in items:
            if item["role"] == "user" and (item["content"] or "").strip():
                title = derive_title(item["content"])
                break
        stamp = _now()
        cursor = conn.execute(
            "INSERT INTO conversations(title, model, created_at, updated_at) VALUES (?,?,?,?)",
            (title, None, stamp, stamp),
        )
        conv_id = cursor.lastrowid
        for item in items:
            role = "assistant" if item["role"] in ("bot", "assistant") else "user"
            content = item["content"] or ""
            if not content.strip():
                continue
            conn.execute(
                "INSERT INTO messages(conversation_id, role, content, attachments, created_at)"
                " VALUES (?,?,?,?,?)",
                (conv_id, role, content, None, stamp),
            )
    conn.execute("INSERT INTO meta(key, value) VALUES('legacy_imported', '1')")
    conn.commit()


def derive_title(text: str, limit: int = 44) -> str:
    clean = " ".join((text or "").split())
    if not clean:
        return "New chat"
    if len(clean) <= limit:
        return clean
    return clean[:limit].rstrip() + "..."


def refresh_titles(limit: int = 64) -> int:
    """Name every conversation by its first user request.

    Old rows may carry generic titles. This only rewrites ``conversations.title``
    in the new schema; the legacy ``chat_history`` table is never touched.
    Returns the number of rows updated.
    """
    changed = 0
    for conv in list_conversations():
        try:
            first = next(
                (
                    m.get("content", "")
                    for m in get_messages(conv["id"])
                    if m.get("role") == "user" and str(m.get("content") or "").strip()
                ),
                None,
            )
        except Exception:
            first = None
        if not first:
            continue
        title = derive_title(first, limit)
        if str(conv.get("title") or "") != title:
            set_title(conv["id"], title)
            changed += 1
    return changed


# --------------------------------------------------------------------------- write

def create_conversation(title: str = "New chat", model: Optional[str] = None) -> int:
    stamp = _now()
    conn = connect()
    try:
        cursor = conn.execute(
            "INSERT INTO conversations(title, model, created_at, updated_at) VALUES (?,?,?,?)",
            (title, model, stamp, stamp),
        )
        conn.commit()
        return int(cursor.lastrowid)
    finally:
        conn.close()


def add_message(
    conversation_id: int,
    role: str,
    content: str,
    attachments: Optional[Iterable[str]] = None,
) -> int:
    stamp = _now()
    blob = json.dumps(list(attachments), ensure_ascii=False) if attachments else None
    conn = connect()
    try:
        cursor = conn.execute(
            "INSERT INTO messages(conversation_id, role, content, attachments, created_at)"
            " VALUES (?,?,?,?,?)",
            (conversation_id, role, content, blob, stamp),
        )
        conn.execute(
            "UPDATE conversations SET updated_at=? WHERE id=?", (stamp, conversation_id)
        )
        conn.commit()
        return int(cursor.lastrowid)
    finally:
        conn.close()


def set_title(conversation_id: int, title: str) -> None:
    conn = connect()
    try:
        conn.execute("UPDATE conversations SET title=? WHERE id=?", (title, conversation_id))
        conn.commit()
    finally:
        conn.close()


def set_model(conversation_id: int, model: str) -> None:
    conn = connect()
    try:
        conn.execute("UPDATE conversations SET model=? WHERE id=?", (model, conversation_id))
        conn.commit()
    finally:
        conn.close()


def delete_conversation(conversation_id: int) -> None:
    conn = connect()
    try:
        conn.execute("DELETE FROM messages WHERE conversation_id=?", (conversation_id,))
        conn.execute("DELETE FROM conversations WHERE id=?", (conversation_id,))
        conn.commit()
    finally:
        conn.close()


def delete_message(message_id: int) -> None:
    conn = connect()
    try:
        conn.execute("DELETE FROM messages WHERE id=?", (message_id,))
        conn.commit()
    finally:
        conn.close()


def clear_all() -> None:
    conn = connect()
    try:
        conn.execute("DELETE FROM messages")
        conn.execute("DELETE FROM conversations")
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------- read

def list_conversations() -> List[Dict]:
    conn = connect()
    try:
        rows = conn.execute(
            "SELECT id, title, model, created_at, updated_at FROM conversations"
            " ORDER BY datetime(updated_at) DESC, id DESC"
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_conversation(conversation_id: int) -> Optional[Dict]:
    conn = connect()
    try:
        row = conn.execute(
            "SELECT id, title, model, created_at, updated_at FROM conversations WHERE id=?",
            (conversation_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_messages(conversation_id: int) -> List[Dict]:
    conn = connect()
    try:
        rows = conn.execute(
            "SELECT id, role, content, attachments, created_at FROM messages"
            " WHERE conversation_id=? ORDER BY id",
            (conversation_id,),
        ).fetchall()
    finally:
        conn.close()
    out = []
    for row in rows:
        item = dict(row)
        try:
            item["attachments"] = json.loads(item["attachments"]) if item["attachments"] else []
        except ValueError:
            item["attachments"] = []
        out.append(item)
    return out


def message_count(conversation_id: int) -> int:
    conn = connect()
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM messages WHERE conversation_id=?", (conversation_id,)
        ).fetchone()
        return int(row["n"]) if row else 0
    finally:
        conn.close()
