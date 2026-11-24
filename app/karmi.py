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

from wine_crew import app_data as ad
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


OTHER = "אחר…"


def wine_cards_html(wines: list[dict]) -> str:
    """Small cards, one under the other, straight from the tool result: the name in bold (one
    left-to-right unit), then tags: points and their word, the price in wine red, the signed
    value score. No number here comes from the model's text."""
    cards = "".join(
        '<div class="karmi-wine">'
        f'<div class="karmi-wine-name"><bdi dir="ltr">{html.escape(str(w["title"]))}</bdi></div>'
        '<div class="karmi-tags">'
        f'<span class="karmi-tag">{int(w["points"])} · {html.escape(w["score_label"])}</span>'
        f'<span class="karmi-tag karmi-price">{ad.money(w["price"])}</span>'
        f'<span class="karmi-tag">ציון שווי {ad.signed_html(w["value_score"])}</span>'
        '</div></div>' for w in wines)
    return f'<div class="karmi-cards">{cards}</div>'


def prediction_card_html(block: dict) -> str:
    """The predicted score and its word in large type, what it is for, and the 8-in-10 range."""
    points, _, word = block["score"].partition(": ")
    inputs = block["inputs"]
    about = (f'<bdi dir="ltr">{html.escape(inputs["variety"])}</bdi> · '
             f'{html.escape(ad.COUNTRY_HE.get(inputs["country"], inputs["country"]))} · {ad.money(inputs["price"])}')
    sentence = f'<div class="karmi-predict-range">{html.escape(block["sentence"])}</div>' if block.get("sentence") else ""
    return (f'<div class="karmi-predict"><div class="karmi-predict-for">{about}</div>'
            f'<div class="karmi-predict-score"><span class="karmi-predict-points">{html.escape(points)}</span> '
            f'<span class="karmi-predict-word">{html.escape(word)}</span></div>{sentence}</div>')


def _send(message: str) -> None:
    st.session_state[PENDING] = message


def _send_other(index: int, sends: dict) -> None:
    value = st.session_state.get(f"karmi_other_select_{index}")
    if value:
        _send(sends.get(value, str(value)))


def _choices(block: dict, index: int, active: bool) -> None:
    """Quick-choice buttons (options from the data), and "אחר…" with a searchable menu of all of
    them that also takes typed text. A click sends the option as the user's message. Only the
    latest message's buttons work; older ones stay visible, disabled."""
    with st.container(horizontal=True, key=f"karmi_choices_{index}"):
        for j, option in enumerate(block["options"]):
            st.button(option["label"], key=f"karmi_choice_{index}_{j}", disabled=not active,
                      on_click=_send, args=(option["send"],))
        st.button(OTHER, key=f"karmi_other_{index}", disabled=not active,
                  on_click=st.session_state.__setitem__, args=(f"karmi_other_open_{index}", True))
    if active and st.session_state.get(f"karmi_other_open_{index}"):
        sends = {o["label"]: o["send"] for o in block["all"]}
        st.selectbox("כל האפשרויות", list(sends), index=None, key=f"karmi_other_select_{index}",
                     placeholder="חפשו או הקלידו…", accept_new_options=True,
                     on_change=_send_other, args=(index, sends), label_visibility="collapsed")


def _markdown(text: str) -> None:
    # "$" starts LaTeX in Streamlit markdown ("$20 ... $30" would turn into math).
    st.markdown(som.display_text(text).replace("$", "\\$"))


def _bubble(role: str, text: str, blocks: list | None = None, index: int = -1, active: bool = False,
            item: dict | None = None) -> None:
    """A message. With cards or choices, in this order: the opening, the color note (if any), the
    cards / buttons, then the closing."""
    with st.chat_message(role, avatar=str(FACE) if role == "assistant" else None):
        if not blocks:
            _markdown(text)
            return
        item = item or {}
        _markdown(item.get("intro") or text)
        if item.get("note"):
            _markdown(item["note"])
        for block in blocks:
            if block["type"] == "wines":
                st.html(wine_cards_html(block["wines"]))
            elif block["type"] == "prediction":
                st.html(prediction_card_html(block))
            elif block["type"] == "choices":
                _choices(block, index, active)
        if item.get("outro"):
            _markdown(item["outro"])


# Karmi's 4-second loop (VP9 WebM, no audio) at the top of the panel, over the still image
# (karmi-toast.webp, also its poster). Its <source> matches only without reduced motion, so with
# reduced motion the browser loads no video and shows the still; style.css also hides it. The
# header exists only while the panel is open, so the video is never fetched before that.
VIDEO = ('<video class="karmi-video" autoplay muted loop playsinline aria-hidden="true" '
         'poster="app/static/karmi-toast.webp">'
         '<source src="app/static/karmi-loop.webm" type="video/webm" '
         'media="(prefers-reduced-motion: no-preference)"></video>')


def _header(demo: bool) -> str:
    demo_tag = f'<span class="karmi-demo">{som.DEMO_LABEL}</span>' if demo else ""
    return (f'<div class="karmi-head"><div class="karmi-toast" role="img" aria-label="כרמי מרים כוס יין">'
            f'{VIDEO}</div>'
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
        last = len(state["shown"]) - 1
        for i, item in enumerate(state["shown"]):
            _bubble(item["role"], item["text"], item["blocks"], i, active=i == last and not pending, item=item)
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
        if chat.open:
            # Marker: while the panel is open the button stands still (no float, no ring).
            st.html('<span class="karmi-open" aria-hidden="true"></span>')
    if chat.open:   # lazy: nothing of the chat (key, openai, tools) runs while it is closed
        with chat, st.container(key="karmi_panel"):
            _panel(load_data, fingerprint, llm, env_file)
