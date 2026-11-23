"""Turning uploaded files into something an OpenRouter chat request can carry.

Two shapes come out of here:

* text   - PDF / plain-text documents, flattened to a string and prepended to the
           user prompt as labelled context.
* image  - actual images, base64 data URLs, sent as OpenAI-style `image_url`
           content parts. Only vision-capable models accept these.

Nothing is written to disk: the bytes live in the Streamlit session only.
"""

from __future__ import annotations

import base64
import io
import mimetypes
from typing import Dict, List, Optional

MAX_TEXT_CHARS = 60_000  # per file, keeps a stray 500-page PDF from nuking the context
MAX_TOTAL_CHARS = 120_000
MAX_IMAGE_BYTES = 8 * 1024 * 1024

TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".log",
    ".py", ".js", ".ts", ".html", ".htm", ".xml", ".yaml", ".yml",
    ".ini", ".conf", ".cfg", ".sh", ".sql", ".c", ".h", ".java", ".go",
}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

# Streamlit's file_uploader wants extensions without the dot.
ACCEPTED_TYPES = sorted(e.lstrip(".") for e in TEXT_EXTENSIONS | IMAGE_EXTENSIONS | {".pdf"})


def _ext(name: str) -> str:
    name = (name or "").lower()
    dot = name.rfind(".")
    return name[dot:] if dot >= 0 else ""


def _pdf_to_text(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    chunks = []
    for number, page in enumerate(reader.pages, 1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # malformed page should not kill the whole upload
            text = f"[page {number} unreadable: {exc}]"
        if text.strip():
            chunks.append(f"--- page {number} ---\n{text.strip()}")
    return "\n\n".join(chunks)


def _image_data_url(data: bytes, name: str) -> str:
    mime = mimetypes.guess_type(name)[0] or "image/png"
    if mime == "image/jpg":
        mime = "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def process_upload(name: str, data: bytes) -> Dict:
    """Classify one uploaded file and return what the chat layer needs."""
    ext = _ext(name)
    result: Dict = {"name": name, "size": len(data), "ext": ext, "kind": None, "error": None}

    if ext in IMAGE_EXTENSIONS:
        if len(data) > MAX_IMAGE_BYTES:
            result["error"] = f"{name}: {len(data) / 1e6:.1f} MB exceeds the 8 MB image limit"
            return result
        result["kind"] = "image"
        result["data_url"] = _image_data_url(data, name)
        return result

    if ext == ".pdf":
        try:
            text = _pdf_to_text(data)
        except Exception as exc:
            result["error"] = f"{name}: could not read PDF ({exc})"
            return result
        if not text.strip():
            result["error"] = f"{name}: no selectable text found (scanned/image-only PDF?)"
            return result
    elif ext in TEXT_EXTENSIONS:
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("latin-1", errors="replace")
    else:
        result["error"] = f"{name}: unsupported type '{ext or 'no extension'}'"
        return result

    truncated = len(text) > MAX_TEXT_CHARS
    if truncated:
        text = text[:MAX_TEXT_CHARS] + f"\n\n[truncated: showing first {MAX_TEXT_CHARS} characters]"
    result["kind"] = "text"
    result["text"] = text
    result["chars"] = len(text)
    result["truncated"] = truncated
    return result


def build_context(attachments: List[Dict]) -> str:
    """Flatten text attachments into one labelled block for the prompt."""
    texts = [a for a in attachments if a.get("kind") == "text" and not a.get("error")]
    if not texts:
        return ""
    parts = []
    total = 0
    for item in texts:
        body = item["text"]
        if total + len(body) > MAX_TOTAL_CHARS:
            body = body[: max(0, MAX_TOTAL_CHARS - total)]
            body += "\n[truncated: total attachment budget reached]"
        total += len(body)
        parts.append(f"===== FILE: {item['name']} =====\n{body}")
    return "\n\n".join(parts)


def image_attachments(attachments: List[Dict]) -> List[Dict]:
    return [a for a in attachments if a.get("kind") == "image" and not a.get("error")]


def summarise(attachments: List[Dict]) -> str:
    """One-line description shown under the uploader."""
    if not attachments:
        return ""
    images = len(image_attachments(attachments))
    texts = len([a for a in attachments if a.get("kind") == "text" and not a.get("error")])
    errors = len([a for a in attachments if a.get("error")])
    bits = []
    if texts:
        chars = sum(a.get("chars", 0) for a in attachments if a.get("kind") == "text")
        bits.append(f"{texts} document(s), {chars:,} chars")
    if images:
        bits.append(f"{images} image(s)")
    if errors:
        bits.append(f"{errors} rejected")
    return " · ".join(bits)
