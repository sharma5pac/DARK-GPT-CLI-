"""Landing script: DarkGPT opens straight into the chat, like ChatGPT does."""

import streamlit as st

st.set_page_config(
    page_title="DarkGPT",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.switch_page("pages/DarkGPT.py")
