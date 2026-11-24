"""DarkGPT chat - ChatGPT-style UI over OpenRouter.

Behaviour that matters:

* Every turn replays the stored transcript, so the model remembers the thread.
* Replies stream token by token.
* Threads live in SQLite (utils/history.py): sidebar list, rename by title,
  per-thread delete, full-text history on reload.
* Attachments (PDF / text / images) are per-message, not per-session.
"""

import hashlib
import html
import os

import streamlit as st
import streamlit.components.v1 as components

from utils import history as db
from utils.attachments import ACCEPTED_TYPES, process_upload, summarise
from utils.get_response import (
    OpenRouterError,
    build_messages,
    build_model_options,
    chat_stream,
    get_api_key,
    get_catalog,
    list_models,
    supports_images,
)

st.set_page_config(
    page_title="DarkGPT",
    page_icon="?",
    layout="centered",
    initial_sidebar_state="expanded",
)

BOT_AVATAR = "?"
USER_AVATAR = "?"
SUGGESTIONS = [
    ("?", "Explain a CVE", "Explain how CVE-2021-44228 (Log4Shell) works and what the fix was."),
    ("?", "Summarise a file", "Summarise the attached document in five bullet points."),
    ("?", "Review code", "Review this code for bugs and security issues, then show a fixed version."),
    ("?", "Explain simply", "Explain how HTTPS actually protects traffic, in plain language."),
]


def local_css(path: str) -> None:
    try:
        with open(path, encoding="utf-8") as handle:
            st.markdown(f"<style>{handle.read()}</style>", unsafe_allow_html=True)
    except OSError:
        pass


local_css("style.css")
db.init_db()


# ------------------------------------------------------------------------- helpers

def current_model() -> str:
    return st.session_state.get("model_id") or "openrouter/free"


def start_new_chat() -> None:
    st.session_state["conversation_id"] = None
    st.session_state["bootstrapped"] = True  # do not auto-reopen the last chat
    st.session_state.pop("attach_cache", None)


def open_chat(conversation_id: int) -> None:
    st.session_state["conversation_id"] = conversation_id
    st.session_state.pop("attach_cache", None)


def short_model(model_id: str) -> str:
    name = model_id.split("/")[-1]
    return name.replace(":free", "")


def autoscroll() -> None:
    """Keep the newest message in view, like a chat app."""
    components.html(
        """
        <script>
        const view = window.parent.document.querySelector('[data-testid="stAppViewContainer"] > section')
                   || window.parent.document.querySelector('section.main');
        if (view) { view.scrollTop = view.scrollHeight; }
        </script>
        """,
        height=0,
    )


# ------------------------------------------------------------------------- sidebar

def sidebar_model() -> str:
    include_uncensored = st.sidebar.toggle(
        "Uncensored models",
        value=bool(st.session_state.get("include_uncensored_models", False)),
        help=(
            "Adds models that are uncensored by design (Dolphin/Venice, TheDrummer, Euryale, "
            "MythoMax). None are free - roughly $0.08-$3 per million tokens. Needs credit on "
            "the OpenRouter account."
        ),
        key="include_uncensored_models",
    )

    models = list_models(include_uncensored=include_uncensored)
    options = build_model_options(models)
    labels = list(options)
    if not labels:
        return "openrouter/free"

    remember = st.session_state.get("model_id")
    index = next((i for i, label in enumerate(labels) if options[label] == remember), 0)
    choice = st.sidebar.selectbox(
        "Model",
        labels,
        index=index,
        help="Models marked ? images accept pictures. Prices are per million tokens.",
    )
    st.session_state["model_id"] = options[choice]

    free_count = len(get_catalog()["free"])
    paid = len(models) - free_count
    caption = f"{free_count} free"
    if paid:
        caption += f" | {paid} paid"
    st.sidebar.caption(caption + " | via OpenRouter")
    return options[choice]


def sidebar_conversations() -> None:
    active = st.session_state.get("conversation_id")
    rows = db.list_conversations()

    st.sidebar.markdown(
        "<div class='dg-side-label'>Chats</div>", unsafe_allow_html=True
    )
    if not rows:
        st.sidebar.caption("Nothing yet - say hello.")
        return

    for conv in rows:
        title = conv["title"] or "New chat"
        if len(title) > 34:
            title = title[:34] + "..."
        cols = st.sidebar.columns([0.84, 0.16], gap="small")
        if cols[0].button(
            ("? " if conv["id"] == active else "  ") + title,
            key=f"open_{conv['id']}",
            use_container_width=True,
        ):
            open_chat(conv["id"])
            st.rerun()
        if cols[1].button("?", key=f"del_{conv['id']}", help="Delete this chat"):
            db.delete_conversation(conv["id"])
            if conv["id"] == active:
                start_new_chat()
            st.rerun()


def sidebar_settings(api_key_present: bool) -> str | None:
    with st.sidebar.expander("[*]? Settings", expanded=not api_key_present):
        pasted = ""
        if api_key_present:
            st.success("OpenRouter key loaded", icon="?")
        else:
            st.warning("No OpenRouter key", icon="?")
            pasted = st.text_input(
                "OpenRouter API key",
                type="password",
                help="Free key from https://openrouter.ai/keys - no card needed.",
                key="openrouter_key_input",
            )
            st.caption("Free tier: ~20 requests/min, 50/day per account.")

        st.markdown("---")
        st.caption("Danger zone")
        confirm = st.checkbox("Allow deleting every chat", key="confirm_clear")
        if st.button("Clear all chats", disabled=not confirm, use_container_width=True):
            db.clear_all()
            start_new_chat()
            st.rerun()
    return (pasted or "").strip() or None


# ------------------------------------------------------------------------ composer

def collect_attachments(model_id: str) -> list:
    """Uploader + per-file parsing, cached by content hash across reruns."""
    label = "? Attach files"
    files = st.file_uploader(
        label,
        type=ACCEPTED_TYPES,
        accept_multiple_files=True,
        help=(
            "PDF, text, CSV, JSON or code -> sent as context. Images -> sent as images and "
            "need a model marked ?. Nothing is written to disk."
        ),
        key="attachments_uploader",
        label_visibility="collapsed",
    )
    if not files:
        st.session_state.pop("attach_cache", None)
        return []

    cache = st.session_state.get("attach_cache", {})
    fresh, processed = {}, []
    for handle in files:
        data = handle.getvalue()
        key = f"{handle.name}:{hashlib.sha1(data).hexdigest()}"
        if key not in cache:
            cache[key] = process_upload(handle.name, data)
        fresh[key] = cache[key]
        processed.append(cache[key])
    st.session_state["attach_cache"] = fresh

    bad = [item for item in processed if item.get("error")]
    for item in bad:
        st.error(item["error"])
    good = [item for item in processed if not item.get("error")]
    if good:
        st.caption(f"Attached: {summarise(good)}")
        images = [item for item in good if item.get("kind") == "image"]
        if images and not supports_images(model_id):
            st.warning(
                f"{short_model(model_id)} cannot see images - pick a model marked ? images."
            )
    return good


# ---------------------------------------------------------------------------- chat

def render_messages(messages: list) -> None:
    for message in messages:
        role = "assistant" if message["role"] in ("assistant", "bot") else "user"
        with st.chat_message(role, avatar=BOT_AVATAR if role == "assistant" else USER_AVATAR):
            st.markdown(message["content"])
            if message.get("attachments"):
                st.caption("? " + ", ".join(message["attachments"]))

    # Controls under the last turn.
    if not messages:
        return
    last = messages[-1]
    if last["role"] == "user":
        if st.button("? Try again", key=f"retry_{last['id']}"):
            st.session_state["pending_retry"] = last["id"]
            st.rerun()
    elif last["role"] in ("assistant", "bot") and len(messages) > 1:
        if st.button("? Regenerate", key=f"regen_{last['id']}"):
            db.delete_message(last["id"])
            previous = messages[-2]
            if previous["role"] == "user":
                st.session_state["pending_retry"] = previous["id"]
            st.rerun()


def stream_reply(api_messages: list, model: str, api_key: str | None) -> str | None:
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        try:
            return st.write_stream(chat_stream(api_messages, model=model, api_key=api_key))
        except OpenRouterError as exc:
            st.error(str(exc))
        except Exception as exc:  # never leave the UI silent
            st.error(f"Unexpected error: {exc}")
    return None


def send(prompt: str, attachments: list, model: str, api_key: str | None) -> None:
    conversation_id = st.session_state.get("conversation_id")
    if not conversation_id:
        conversation_id = db.create_conversation(db.derive_title(prompt), model)
        st.session_state["conversation_id"] = conversation_id

    history = db.get_messages(conversation_id)  # everything before this turn
    names = [item["name"] for item in attachments]
    db.add_message(conversation_id, "user", prompt, names)
    if not history:
        db.set_title(conversation_id, db.derive_title(prompt))
    db.set_model(conversation_id, model)

    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)
        if names:
            st.caption("? " + ", ".join(names))

    api_messages = build_messages(history=history, prompt=prompt, attachments=attachments)
    answer = stream_reply(api_messages, model, api_key)

    if answer and answer.strip():
        db.add_message(conversation_id, "assistant", answer.strip())
        st.rerun()
    else:
        st.session_state.pop("pending_retry", None)
    autoscroll()


def retry(message_id: int, model: str, api_key: str | None) -> None:
    conversation_id = st.session_state.get("conversation_id")
    if not conversation_id:
        return
    messages = db.get_messages(conversation_id)
    target = next((m for m in messages if m["id"] == message_id), None)
    if not target or target["role"] != "user":
        return
    history = [m for m in messages if m["id"] < message_id]
    api_messages = build_messages(history=history, prompt=target["content"])
    answer = stream_reply(api_messages, model, api_key)
    if answer and answer.strip():
        db.add_message(conversation_id, "assistant", answer.strip())
    st.session_state.pop("pending_retry", None)
    st.rerun()


# ---------------------------------------------------------------------------- main

def main() -> None:
    st.session_state.setdefault("conversation_id", None)
    st.session_state.setdefault("model_id", "openrouter/free")

    key_present = bool(get_api_key())
    api_key = sidebar_settings(key_present)
    model = sidebar_model()

    # Brand + new chat
    st.sidebar.markdown(
        "<div class='dg-brand'><span class='dg-brand-mark'>?</span> DarkGPT</div>",
        unsafe_allow_html=True,
    )
    if st.sidebar.button("?  New chat", use_container_width=True, type="primary"):
        start_new_chat()
        st.rerun()
    sidebar_conversations()

    # ChatGPT behaviour: opening the app lands you back in the last thread.
    if st.session_state.get("conversation_id") is None and not st.session_state.get("bootstrapped"):
        recent = db.list_conversations()
        if recent:
            st.session_state["conversation_id"] = recent[0]["id"]
        st.session_state["bootstrapped"] = True

    conversation_id = st.session_state.get("conversation_id")
    conversation = db.get_conversation(conversation_id) if conversation_id else None
    messages = db.get_messages(conversation_id) if conversation_id else []

    # Header
    title = conversation["title"] if conversation else "New chat"
    st.markdown(
        f"<div class='dg-header'><span class='dg-title'>{html.escape(title)}</span>"
        f"<span class='dg-badge'>{html.escape(short_model(model))}</span></div>",
        unsafe_allow_html=True,
    )

    if messages:
        render_messages(messages)
    else:
        st.markdown(
            "<div class='dg-hero'><div class='dg-hero-mark'>?</div>"
            "<h1>How can I help?</h1>"
            "<p>Ask anything. Attach a PDF, a file or an image below. "
            "The thread is remembered, and it is saved when you come back.</p></div>",
            unsafe_allow_html=True,
        )
        cols = st.columns(2)
        for index, (icon, label, text) in enumerate(SUGGESTIONS):
            with cols[index % 2]:
                if st.button(f"{icon}  {label}", key=f"suggest_{index}", use_container_width=True):
                    st.session_state["pending_prompt"] = text
                    st.rerun()

    # Retry a failed turn, or regenerate after the last answer was deleted.
    pending_retry = st.session_state.pop("pending_retry", None)
    if pending_retry:
        retry(pending_retry, model, api_key)

    attachments = collect_attachments(model)
    prompt = st.chat_input("Message DarkGPT...")
    pending_prompt = st.session_state.pop("pending_prompt", None)
    prompt = prompt or pending_prompt

    if prompt:
        send(prompt, attachments, model, api_key)

    st.markdown(
        "<div class='dg-footnote'>DarkGPT runs on OpenRouter. Free models may use your "
        "prompts for training - do not paste client or personal data.",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
