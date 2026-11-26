"""HTTP + SSE backend for the React chat UI.

Reuses the existing pieces rather than reimplementing them:

    utils.get_response  -> model catalogue, message builder, OpenRouter streaming
    utils.attachments   -> PDF / text / image handling
    utils.history       -> the same chat_history.db the Streamlit page uses

    GET    /api/models
    GET    /api/conversations
    POST   /api/conversations
    DELETE /api/conversations/{id}
    GET    /api/conversations/{id}/messages
    POST   /api/messages/{id}/delete
    POST   /api/chat                 (multipart, streams text/event-stream)
    GET    /uploads/<file>           (images attached to earlier messages)

Run:  uvicorn server.api:app --host 127.0.0.1 --port 8502
"""

from __future__ import annotations

import hashlib
import json
import os
import base64
import mimetypes
from typing import Dict, Iterable, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from utils import history as db
from utils.attachments import process_upload
from utils.get_response import (
    OpenRouterError,
    build_messages,
    chat_stream,
    get_api_key,
    get_catalog,
    list_models,
)
from utils.search import format_results, web_search
from utils.agent import DEFAULT_AGENT_MODEL, agent_stream

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(PROJECT_ROOT, ".uploads")
DIST_DIR = os.path.join(PROJECT_ROOT, "web", "dist")

os.makedirs(UPLOAD_DIR, exist_ok=True)
db.init_db()

app = FastAPI(title="DarkGPT", docs_url=None, redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- utils

def sse(payload: Dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _store_image(name: str, data: bytes) -> Optional[str]:
    ext = os.path.splitext(name)[1].lower() or ".png"
    digest = hashlib.sha1(data).hexdigest()
    filename = f"{digest}{ext}"
    path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(path):
        with open(path, "wb") as handle:
            handle.write(data)
    return f"/uploads/{filename}"


def _attachment_records(processed: Iterable[Dict], raw: Dict[str, bytes]) -> List[Dict]:
    records: List[Dict] = []
    for item in processed:
        if item.get("error"):
            continue
        record = {"name": item["name"], "kind": item.get("kind")}
        if item.get("kind") == "image":
            data = raw.get(item["name"])
            if data:
                record["url"] = _store_image(item["name"], data)
        elif item.get("kind") == "text" and item.get("text"):
            # Persist extracted text so follow-up turns can still reference the file.
            record["text"] = item["text"]
        records.append(record)
    return records


def _normalise_attachments(stored) -> List[Dict]:
    out = []
    for item in stored or []:
        if isinstance(item, str):
            out.append({"name": item, "kind": "text"})
        elif isinstance(item, dict):
            out.append(item)
    return out


MAX_HISTORY_IMAGES = 4
MAX_HISTORY_IMAGE_BYTES = 4 * 1024 * 1024


def _file_to_data_url(path: str) -> Optional[str]:
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError:
        return None
    if not data or len(data) > MAX_HISTORY_IMAGE_BYTES:
        return None
    mime = mimetypes.guess_type(path)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def hydrate_history_attachments(messages: List[Dict]) -> List[Dict]:
    """Load stored attachment content so follow-up turns can see old files/images."""
    hydrated = []
    # Keep only the most recent N images across history to cap payload size.
    image_positions = []
    for i, msg in enumerate(messages):
        for a in _normalise_attachments(msg.get("attachments")):
            if a.get("kind") == "image" and a.get("url"):
                image_positions.append((i, a.get("url")))
    keep = set(image_positions[-MAX_HISTORY_IMAGES:])

    for i, msg in enumerate(messages):
        item = dict(msg)
        raw_atts = _normalise_attachments(msg.get("attachments"))
        atts = []
        for a in raw_atts:
            att = dict(a)
            if att.get("kind") == "image" and att.get("url"):
                if (i, att.get("url")) in keep:
                    url = att["url"]
                    if url.startswith("/uploads/"):
                        path = os.path.join(UPLOAD_DIR, url.split("/uploads/", 1)[1])
                    else:
                        path = os.path.join(PROJECT_ROOT, url.lstrip("/"))
                    data_url = _file_to_data_url(path)
                    if data_url:
                        att["data_url"] = data_url
            atts.append(att)
        item["attachments"] = atts
        hydrated.append(item)
    return hydrated


# ---------------------------------------------------------------------------- api

@app.get("/api/health")
def health() -> Dict:
    return {"ok": True, "key_present": bool(get_api_key())}


@app.get("/api/models")
def models() -> Dict:
    catalog = get_catalog()
    return {
        "free": catalog.get("free", []),
        "uncensored": catalog.get("uncensored", []),
        "default": "cognitivecomputations/dolphin-mistral-24b-venice-edition",
        "key_present": bool(get_api_key()),
    }


@app.get("/api/conversations")
def conversations() -> List[Dict]:
    return db.list_conversations()


@app.post("/api/conversations")
def create_conversation(payload: Optional[Dict] = None) -> Dict:
    title = (payload or {}).get("title") or "New chat"
    model = (payload or {}).get("model")
    conv_id = db.create_conversation(title, model)
    return db.get_conversation(conv_id) or {"id": conv_id, "title": title}


@app.delete("/api/conversations/{conversation_id}")
def delete_conversation(conversation_id: int) -> Dict:
    db.delete_conversation(conversation_id)
    return {"ok": True}


@app.get("/api/conversations/{conversation_id}/messages")
def messages(conversation_id: int) -> List[Dict]:
    rows = db.get_messages(conversation_id)
    for row in rows:
        row["attachments"] = _normalise_attachments(row.get("attachments"))
    return rows


@app.post("/api/messages/{message_id}/delete")
def delete_message(message_id: int) -> Dict:
    db.delete_message(message_id)
    return {"ok": True}


@app.post("/api/chat")
def chat(
    message: str = Form(""),
    model: str = Form("cognitivecomputations/dolphin-mistral-24b-venice-edition"),
    conversation_id: Optional[int] = Form(None),
    resume: bool = Form(False),
    web_search_enabled: bool = Form(False),
    files: List[UploadFile] = File(default=[]),
):
    """Stream one assistant reply. `resume` re-answers the last user turn."""
    api_key = None  # resolved from .env / environment inside get_response

    raw: Dict[str, bytes] = {}
    processed: List[Dict] = []
    for upload in files or []:
        data = upload.file.read()
        if not data:
            continue
        raw[upload.filename] = data
        processed.append(process_upload(upload.filename, data))

    failures = [item["error"] for item in processed if item.get("error")]
    good = [item for item in processed if not item.get("error")]

    if resume:
        if not conversation_id:
            raise HTTPException(400, "resume needs a conversation_id")
        messages_all = hydrate_history_attachments(db.get_messages(conversation_id))
        last_user = next((m for m in reversed(messages_all) if m["role"] == "user"), None)
        if last_user is None:
            raise HTTPException(400, "nothing to resume")
        history = [m for m in messages_all if m["id"] < last_user["id"]]
        prompt = last_user["content"]
        good = _normalise_attachments(last_user.get("attachments"))
    else:
        prompt = (message or "").strip()
        if not prompt and not good:
            raise HTTPException(400, "empty message")
        if not conversation_id:
            conversation_id = db.create_conversation(db.derive_title(prompt or "Attachment"), model)
        elif db.get_conversation(conversation_id) is None:
            conversation_id = db.create_conversation(db.derive_title(prompt or "Attachment"), model)
        history = hydrate_history_attachments(db.get_messages(conversation_id))
        records = _attachment_records(good, raw)
        db.add_message(conversation_id, "user", prompt, records)
        if not history:
            db.set_title(conversation_id, db.derive_title(prompt or "Attachment"))
        db.set_model(conversation_id, model)

    sources: List[Dict[str, str]] = []
    model_prompt = prompt
    if web_search_enabled and not resume and prompt.strip():
        sources = web_search(prompt, max_results=5)
        ctx = format_results(sources)
        if ctx:
            model_prompt = f"{ctx}\n\n===== USER QUESTION =====\n{prompt}"

    api_messages = build_messages(history=history, prompt=model_prompt, attachments=good)
    conversation = db.get_conversation(conversation_id) or {}

    def generate():
        accumulator: List[str] = []
        saved = False
        try:
            yield sse(
                {
                    "type": "meta",
                    "conversation_id": conversation_id,
                    "title": conversation.get("title"),
                    "model": model,
                    "sources": sources,
                }
            )
            for failure in failures:
                yield sse({"type": "warning", "message": failure})
            for chunk in chat_stream(api_messages, model=model, api_key=api_key):
                accumulator.append(chunk)
                yield sse({"type": "delta", "text": chunk})
            answer = "".join(accumulator).strip()
            if answer:
                db.add_message(conversation_id, "assistant", answer)
                saved = True
            yield sse({"type": "done", "model": model})
        except OpenRouterError as exc:
            yield sse({"type": "error", "message": str(exc)})
        except Exception as exc:  # noqa: BLE001 - must not kill the stream silently
            yield sse({"type": "error", "message": f"Unexpected error: {exc}"})
        finally:
            partial = "".join(accumulator).strip()
            if partial and not saved:
                db.add_message(conversation_id, "assistant", partial)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/agent")
def agent(
    message: str = Form(""),
    model: str = Form(DEFAULT_AGENT_MODEL),
    conversation_id: Optional[int] = Form(None),
):
    prompt = (message or "").strip()
    if not prompt:
        raise HTTPException(400, "empty message")
    if not conversation_id:
        conversation_id = db.create_conversation(db.derive_title(prompt), model)
    elif db.get_conversation(conversation_id) is None:
        conversation_id = db.create_conversation(db.derive_title(prompt), model)

    history = hydrate_history_attachments(db.get_messages(conversation_id))
    db.add_message(conversation_id, "user", prompt)
    if not history:
        db.set_title(conversation_id, db.derive_title(prompt))
    db.set_model(conversation_id, model)
    conversation = db.get_conversation(conversation_id) or {}

    def generate():
        text_parts: List[str] = []
        saved = False
        try:
            yield sse({"type": "meta", "conversation_id": conversation_id, "title": conversation.get("title"), "model": model, "agent": True})
            for event in agent_stream(prompt, history=history, model=model):
                if event.get("type") == "delta":
                    text_parts.append(event.get("text") or "")
                if event.get("type") == "autonomous_plan":
                    # Explicit web parity: keep the same shape the TUI consumes.
                    event = {
                        "type": "autonomous_plan",
                        "goal": str(event.get("goal") or ""),
                        "steps": [dict(s) for s in (event.get("steps") or []) if isinstance(s, dict)],
                    }
                yield sse(event)
                if event.get("type") in ("done", "error"):
                    break
            answer = "".join(text_parts).strip()
            if answer:
                db.add_message(conversation_id, "assistant", answer)
                saved = True
        except Exception as exc:  # noqa: BLE001
            yield sse({"type": "error", "message": str(exc)})
        finally:
            partial = "".join(text_parts).strip()
            if partial and not saved:
                db.add_message(conversation_id, "assistant", partial)

    return StreamingResponse(generate(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ------------------------------------------------------------------------ statics

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

if os.path.isdir(DIST_DIR):
    assets_dir = os.path.join(DIST_DIR, "assets")
    if os.path.isdir(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    index_path = os.path.join(DIST_DIR, "index.html")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(index_path)

    @app.get("/{path:path}")
    def spa_fallback(path: str) -> FileResponse:
        # Don't swallow API/uploads misses - they should 404 properly.
        if path.startswith("api/") or path.startswith("uploads/"):
            raise HTTPException(404, "Not found")
        target = os.path.join(DIST_DIR, path)
        if os.path.isfile(target):
            return FileResponse(target)
        return FileResponse(index_path)
else:  # pragma: no cover - only before the first build

    @app.get("/")
    def placeholder():
        raise HTTPException(503, "web/dist not built yet - run npm run build in web/")
