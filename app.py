import streamlit as st
from rag import (
    load_index,
    retrieve,
    build_context,
    generate_answer,
    refuse,
    MIN_SCORE,
    should_refuse_query,
)

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Existentialism Corpus Agent", layout="wide")

st.markdown(
    """
    <style>
    /* Expander container */
    div[data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.15);
    }

    /* Expander header text */
    div[data-testid="stExpander"] summary {
        color: #ffffff !important;
        font-weight: 500;
        font-size: 15px;
    }

    /* Expander body text (THIS is the important part) */
    div[data-testid="stExpander"] div {
        color: #ffffff !important;
        font-size: 14px;
        line-height: 1.6;
    }

    /* Code/text blocks inside expander */
    pre, code, textarea {
        color: #ffffff !important;
        background: transparent !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
    /* Make all input labels visible */
    label, .stTextInput label {
        color: #ffffff !important;
        font-weight: 600;
    }

    /* Make placeholder text clearer */
    input::placeholder {
        color: #cbd5e1 !important;
        opacity: 1;
    }

    /* Keep typed text white */
    input {
        color: #ffffff !important;
        background-color: rgba(255,255,255,0.08) !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------- GLOBAL STYLING --------------------
st.markdown(
    """
    <style>
    /* Remove white bar at top */
    header {visibility: hidden;}

    /* Page background */
    .stApp {
        background: radial-gradient(circle at top left,
            #1e293b,
            #020617 60%);
        color: #e5e7eb;
    }

    /* Titles */
    h1, h2, h3 {
        color: #f8fafc;
    }

    /* Input box */
    input {
        background-color: rgba(15, 23, 42, 0.95) !important;
        color: #ffffff !important;
        border-radius: 10px;
        border: 1px solid rgba(148, 163, 184, 0.3);
        font-size: 1rem;
    }

    /* Placeholder */
    input::placeholder {
        color: #94a3b8 !important;
        opacity: 1;
    }

    /* Keep text white on focus */
    input:focus {
        color: #ffffff !important;
    }

    /* Answer box */
    .answer-box {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-radius: 14px;
        padding: 1.2rem;
        box-shadow: 0 0 30px rgba(56, 189, 248, 0.08);
    }

    /* Sidebar / sources */
    .source-box {
        background: rgba(30, 41, 59, 0.75);
        border-radius: 12px;
        padding: 1rem;
    }

    /* Buttons */
    button {
        background: linear-gradient(135deg, #0ea5e9, #38bdf8) !important;
        color: #020617 !important;
        border-radius: 10px !important;
        border: none !important;
        font-weight: 600 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    /* Expander title */
    div[data-testid="stExpander"] summary,
    div[data-testid="stExpander"] summary * {
        color: #ffffff !important;
        opacity: 1 !important;
    }

    /* Expander body (THIS is the key) */
    div[data-testid="stExpanderDetails"] * {
        color: #ffffff !important;
        opacity: 1 !important;
    }

    /* Specifically st.text() output */
    div[data-testid="stExpanderDetails"] div[data-testid="stText"] pre {
        color: #ffffff !important;
        opacity: 1 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------- HEADER --------------------
st.title("📚 Existentialism Corpus Agent")
st.caption(
    "Answers ONLY using the uploaded PDFs. Refuses when the corpus doesn’t support an answer."
)

# -------------------- LOAD INDEX --------------------
@st.cache_resource
def init():
    return load_index()

index, chunks, model = init()

# -------------------- INPUT (SESSION SAFE) --------------------
query = st.text_input(
    "Ask a question:",
    placeholder="e.g., What does Sartre mean by abandonment?",
    key="query",
)

# -------------------- MAIN LOGIC --------------------
if st.button("Ask") and query.strip():

    # Gate 1 — forbidden topics
    if should_refuse_query(query):
        st.error(refuse())
        st.stop()

    passages, best = retrieve(query, index, chunks, model)

    col1, col2 = st.columns([2, 1])

    # Right column — retrieval info
    with col2:
        st.subheader("Retrieval score")
        st.write(f"Best score: **{best:.3f}**")
        st.write(f"Threshold: **{MIN_SCORE:.3f}**")

        if best < MIN_SCORE or len(passages) == 0:
            st.error(refuse())
            st.stop()

    # Gate 2 — literal support check
    if should_refuse_query(query, passages=passages):
        st.error(refuse())
        st.stop()

    context = build_context(passages)
    answer = generate_answer(query, context)

    # Gate 3 — model self-refusal
    if answer.strip() == refuse().strip():
        st.error(refuse())
        st.stop()

    # -------------------- ANSWER --------------------
    with col1:
        st.markdown(
            f"""
            <div class="answer-box">
            {answer}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -------------------- SOURCES --------------------
    with col2:
        st.subheader("Sources used")
        st.markdown('<div class="source-box">', unsafe_allow_html=True)
        for p in passages:
            st.write(
                f"- {p['source']} p.{p['page']} (score {p['score']:.3f})"
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------- CONTEXT --------------------
    with st.expander("Show retrieved context"):
        st.text(context)