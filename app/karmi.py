"""Karmi (כרמי) in the app: the floating button in the bottom-left corner of every tab and the
chat panel it opens (a st.popover, which stays open while the user types; checked in Streamlit
1.67 at 1280 and 375 px). The agent itself is wine_crew.sommelier.

Nothing runs until the panel is opened: openai is imported and the key is read (st.secrets, the
environment, then a local .env) only then. The conversation lives in st.session_state only; the
one shared thing is an in-memory cache of first-question answers (st.cache_data, no disk).
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Callable

import streamlit as st

from wine_crew import sommelier as som

STATIC = Path(__file__).resolve().parent / "static"
FACE = STATIC / "karmi-face.webp"          # the button (via style.css) and Karmi's chat avatar
STATE = "karmi_state"
OPENED = "karmi_opened"
PENDING = "karmi_pending"
TYPING = ('<div class="karmi-typing" role="status" aria-label="כרמי מקליד">'
          '<span></span><span></span><span></span></div>')


class _Uncached(Exception):
    """Carries an answer that failed the number check out of the cached function, so it is
    shown but not cached (st.cache_data does not store a call that raised)."""

    def __init__(self, reply: som.Reply):
        super().__init__("not cached")
        self.reply = reply


@st.cache_data(max_entries=som.CACHE_MAX_ENTRIES, ttl=som.CACHE_TTL_SECONDS, show_spinner=False)
def _first_answer(question_key: str, fingerprint: tuple, model: str, _run: Callable[[], som.Reply]) -> som.Reply:  # noqa: ARG001
    """The answer to a conversation's first question. Key: the normalized question, the
    artifacts' fingerprint (new artifacts -> a new key) and the model ("demo" in demo mode).
    In memory only, max_entries and ttl from sommelier."""
    reply = _run()
    if not reply.ok:
        raise _Uncached(reply)
    return reply


def first_answer_cache(fingerprint: tuple, model: str) -> Callable[[str, Callable[[], som.Reply]], som.Reply]:
    def cache(question_key: str, run: Callable[[], som.Reply]) -> som.Reply:
        try:
            return _first_answer(question_key, fingerprint, model, run)
        except _Uncached as exc:
            return exc.reply
    return cache


def _bubble(role: str, text: str) -> None:
    with st.chat_message(role, avatar=str(FACE) if role == "assistant" else None):
        # "$" starts LaTeX in Streamlit markdown ("$20 ... $30" would turn into math).
        st.markdown(som.display_text(text).replace("$", "\\$"))


def _header(demo: bool) -> str:
    # The round image is the place for a short looping video of Karmi later (same size).
    demo_tag = f'<span class="karmi-demo">{som.DEMO_LABEL}</span>' if demo else ""
    return (f'<div class="karmi-head"><div class="karmi-toast" role="img" aria-label="כרמי מרים כוס יין"></div>'
            f'<div class="karmi-head-text"><div class="karmi-title">{html.escape(som.TITLE)}</div>'
            f'<div class="karmi-note">{html.escape(som.DISCLOSURE)}</div>{demo_tag}</div></div>')


def _secret_key():
    return st.secrets.get("OPENAI_API_KEY")


def _panel(load_data: Callable[[], som.Data], fingerprint: tuple, llm: dict, env_file: Path) -> None:
    demo = som.demo_enabled()
    st.html(_header(demo))
    key = None if demo else som.load_api_key(_secret_key, env_file=env_file)
    if not demo and not key:
        st.info(som.NO_KEY)
        return
    state = st.session_state.setdefault(STATE, som.new_state())
    pending = st.session_state.pop(PENDING, None)
    # Before the first question only the welcome is shown, at its natural height (so the sample
    # questions and the input fit on a short screen); then a scrolling box (height in style.css).
    history = (st.container(height=340, key="karmi_history") if state["shown"]
               else st.container(key="karmi_welcome"))
    with history:
        _bubble("assistant", som.WELCOME)
        for role, text in state["shown"]:
            _bubble(role, text)
    if not state["shown"] and not pending:
        with st.container(key="karmi_samples"):
            for i, sample in enumerate(som.SAMPLE_QUESTIONS):
                st.button(sample, key=f"karmi_sample_{i}", width="stretch",
                          on_click=st.session_state.__setitem__, args=(PENDING, sample))
    left = som.remaining(state)
    asked = st.chat_input(som.PLACEHOLDER, key="karmi_input", disabled=left == 0, max_chars=300)
    if state["count"]:
        st.caption(f"נשארו {left} שאלות בשיחה הזו." if left else som.LIMIT_REACHED)
    question = (pending or asked or "").strip()
    if not question:
        return
    with history:
        _bubble("user", question)
        with st.chat_message("assistant", avatar=str(FACE)):
            st.html(TYPING)
    model = "demo" if demo else llm["model"]
    som.handle_question(
        state, question, client_factory=lambda: som.make_client(key), tools=som.Tools(load_data()),
        model=model, temperature=float(llm.get("temperature", 0)),
        cache=first_answer_cache(fingerprint, model))
    st.rerun()   # show the answer in place of the typing dots (the popover stays open)


def render(load_data: Callable[[], som.Data], fingerprint: tuple, llm: dict, env_file: Path) -> None:
    """The floating button and, while it is open, the chat panel. Outside the tabs, so on every tab."""
    with st.container(key="karmi"):
        chat = st.popover(som.HOVER, key="karmi_chat", on_change="rerun", help=som.HOVER)
        if chat.open:
            st.session_state[OPENED] = True
        if not st.session_state.get(OPENED):
            # Marker for the gold "invitation" ring in style.css, until the first opening.
            st.html('<span class="karmi-invite" aria-hidden="true"></span>')
    if chat.open:   # lazy: nothing of the chat (key, openai, tools) runs while it is closed
        with chat, st.container(key="karmi_panel"):
            _panel(load_data, fingerprint, llm, env_file)
