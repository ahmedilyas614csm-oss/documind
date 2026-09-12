"""DocuMind — Streamlit UI for a RAG PDF Q&A system with live evaluation.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import os

# Quiet HF noise before any model loads
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from documind.config import get_settings
from documind.pipeline import (
    collection_size,
    ingest_path,
    run_query,
)

# --- Page setup --------------------------------------------------------------

st.set_page_config(
    page_title="DocuMind — RAG Q&A with Evaluation",
    page_icon="🧠",
    layout="wide",
)

settings = get_settings()

if "history" not in st.session_state:
    st.session_state.history = []  # list of QueryResult

# --- Sidebar -----------------------------------------------------------------

with st.sidebar:
    st.title("🧠 DocuMind")
    st.caption("RAG PDF Q&A with live evaluation")

    st.divider()
    st.subheader("1. Ingest documents")

    uploaded = st.file_uploader(
        "Drop a PDF here",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded:
        upload_dir = Path("data/uploads")
        upload_dir.mkdir(parents=True, exist_ok=True)
        for f in uploaded:
            dest = upload_dir / f.name
            dest.write_bytes(f.getbuffer())
            with st.spinner(f"Ingesting {f.name}..."):
                n = ingest_path(dest)
            st.success(f"{f.name}: {n} chunks")

    if st.button("Ingest sample PDFs"):
        with st.spinner("Ingesting data/sample_pdfs/..."):
            n = ingest_path("data/sample_pdfs")
        st.success(f"Indexed {n} chunks total")

    st.divider()
    st.subheader("2. Status")
    st.metric("Indexed chunks", collection_size())

    st.divider()
    st.subheader("3. Settings")
    top_k = st.slider("Chunks to retrieve (top-k)", 1, 10, settings.top_k)
    with_eval = st.toggle("Run evaluation metrics", value=True)

    st.divider()
    if st.button("Clear chat history"):
        st.session_state.history = []
        st.rerun()

    st.caption(f"LLM: `{settings.groq_model}`")
    st.caption(f"Judge: `{settings.groq_judge_model}`")

# --- Main --------------------------------------------------------------------

st.title("DocuMind")
st.caption("Ask questions about your indexed PDFs. Get answers with citations and quality metrics.")

if collection_size() == 0:
    st.warning(
        "No documents indexed yet. Upload a PDF or click **Ingest sample PDFs** in the sidebar."
    )
    st.stop()

query = st.chat_input("Ask a question about your documents...")

if query:
    with st.spinner("Retrieving and generating..."):
        t0 = time.time()
        try:
            result = run_query(query, top_k=top_k, with_evaluation=with_eval)
            elapsed = time.time() - t0
            st.session_state.history.append((query, result, elapsed))
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

# --- Render history ----------------------------------------------------------

for q, result, elapsed in reversed(st.session_state.history):
    with st.chat_message("user"):
        st.write(q)

    with st.chat_message("assistant"):
        st.markdown(result.answer.text)

        # Citations row
        if result.answer.citations:
            st.markdown("**Sources**")
            cols = st.columns(min(4, len(result.answer.citations)))
            for i, c in enumerate(result.answer.citations):
                with cols[i % len(cols)]:
                    st.markdown(
                        f"**[{i + 1}]** `{c['source']}` p.{c['page']}  \n"
                        f"<span style='color:gray'>score {c['score']}</span>",
                        unsafe_allow_html=True,
                    )

        # Metrics
        ev = result.evaluation
        if ev.faithfulness == ev.faithfulness and ev.faithfulness >= 0:  # not NaN, not failed
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Faithfulness", f"{ev.faithfulness:.2f}")
            m2.metric("Answer relevance", f"{ev.answer_relevance:.2f}")
            m3.metric("Context precision", f"{ev.context_precision:.2f}")
            m4.metric("Contexts used", f"{result.contexts_used}/{result.contexts_total}")

        st.caption(f"⏱ {elapsed:.2f}s · judge `{ev.judge_model}`")

        with st.expander("Retrieved context"):
            for i, (cite, text) in enumerate(
                zip(result.answer.citations, result.answer.contexts), 1
            ):
                st.markdown(f"**[{i}] {cite['source']} p.{cite['page']}**  · score {cite['score']}")
                st.text(text[:1500] + ("..." if len(text) > 1500 else ""))
                st.divider()

# --- Dashboard ---------------------------------------------------------------

if st.session_state.history:
    st.divider()
    st.subheader("📊 Evaluation dashboard")

    rows = []
    for q, r, _ in st.session_state.history:
        ev = r.evaluation
        rows.append(
            {
                "Question": q[:60] + ("..." if len(q) > 60 else ""),
                "Faithfulness": ev.faithfulness if ev.faithfulness >= 0 else None,
                "Answer relevance": ev.answer_relevance,
                "Context precision": ev.context_precision,
                "Latency (s)": _,
            }
        )
    df = pd.DataFrame(rows)

    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown("**Per-question metrics**")
        st.dataframe(df, use_container_width=True)

    with col_b:
        st.markdown("**Average scores**")
        avg = df[["Faithfulness", "Answer relevance", "Context precision"]].mean()
        fig = go.Figure(
            go.Bar(
                x=avg.values,
                y=avg.index,
                orientation="h",
                marker=dict(color=["#4C9AFF", "#36B37E", "#FFAB00"]),
                text=[f"{v:.2f}" for v in avg.values],
                textposition="auto",
            )
        )
        fig.update_layout(
            xaxis=dict(range=[0, 1], title="Score"),
            yaxis=dict(title=""),
            showlegend=False,
            height=300,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    if st.button("Download history as CSV"):
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Click to download",
            csv,
            file_name="documind_eval.csv",
            mime="text/csv",
        )
