import os, re, html
from pathlib import Path
import streamlit as st
import pandas as pd
from pipeline import answer_question  # <- now points to your new pipeline
import markdown

# --- Config ---
PDF_BASE_URL = os.getenv("PDF_BASE_URL", "")
st.set_page_config("Bean GPT", "🌱", layout="wide")

# --- Style ---
st.markdown(
    """
    <style>
      #MainMenu, footer {visibility:hidden;}
      .source{background:#1e1f24;padding:.75rem;border-radius:.5rem;margin:.4rem 0;word-wrap:break-word;}
      .source a{color:#6EA8FE;font-weight:600;text-decoration:none;}
    </style>
""",
    unsafe_allow_html=True,
)

# --- Title ---
st.title("🌱 BeanGPT – Main Platform for Dry Beans")

# --- Initialize History ---
if "history" not in st.session_state:
    st.session_state.history = []


# --- Helper Functions ---
def title_from(fname: str) -> str:
    stem = Path(fname).stem
    core = stem.split("_", 1)[-1]
    core = re.sub(r"[_-]+", " ", core)
    return core.title()


def doi_from(source: str) -> str:
    print(f"\n[DEBUG] Raw source input: {source}")
    if "/" in source:
        print(f"[DEBUG] Already a real DOI: {source}")
        return source.strip()
    # Clean `.pdf` and replace the first underscore only
    cleaned = source.strip().replace(".pdf", "")
    doi = cleaned.replace("_", "/", 1)
    print(f"[DEBUG] Parsed to real DOI: {doi}")
    return doi


def link_block(source: str) -> str:
    doi = doi_from(source)
    url = f"https://doi.org/{doi}"
    print(f"[DEBUG] Final DOI link to render: {url}")
    return f'<a href="{url}" target="_blank">🌐 Visit Source</a>'


def render_citation(index: int, doi: str):
    st.markdown(
        f"<div class='source'><b>[{index}] {doi}</b><br>{link_block(doi)}</div>",
        unsafe_allow_html=True,
    )


def show_sources(sources):
    print("\n🔗 [DEBUG] DOIs being rendered:")
    for i, s in enumerate(sources, 1):
        print(f"  → https://doi.org/{s}")
        render_citation(i, s)


def show_genes(gene_list: list[dict]):
    if not gene_list:
        st.info("No gene or protein names were detected in this response.")
        return

    for gene in gene_list:
        st.markdown(f"**🧬 {gene['name']}**", unsafe_allow_html=True)

        lines = (
            gene["summary"].split("\n")
            if isinstance(gene["summary"], str)
            else gene["summary"]
        )
        st.markdown(
            "\n".join([f"- {line.strip('- ')}" for line in lines]),
            unsafe_allow_html=True,
        )
        st.markdown("---")


# --- Wrapper ---
def ask_rag(question: str):
    answer, sources, genes, full_md = answer_question(question)
    return answer, sources, genes, full_md


# --- Render chat history ---
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

        if msg.get("sources") or msg.get("genes"):
            col1, col2 = st.columns([0.6, 0.4])

            if msg.get("sources"):
                with col1.expander("📄 Sources", expanded=True):
                    show_sources(msg["sources"])

            if msg.get("genes") not in (None, [], ""):
                with col2.expander("🧬 Genes Mentioned", expanded=True):
                    show_genes(msg["genes"])


# --- New query handling ---
query = st.chat_input("Ask Bean GPT…")
if query:
    st.session_state.history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.spinner("Thinking…"):
        # Now ask_rag returns (preview_md, sources, genes, full_md)
        answer, sources, genes, full_md = answer_question(query)

    st.session_state.history.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "genes": genes,
        }
    )

    with st.chat_message("assistant"):
        # If the preview contains the “Show more…” marker, render exactly `answer`
        # (10 rows + "**Show more in app to see the full list.**"), then offer a checkbox to show `full_md`.
        if "**Show more in app to see the full list.**" in answer:
            # Render the 10‐row preview + marker
            st.markdown(answer, unsafe_allow_html=False)

            # When checked, render the full table (full_md contains all rows)
            if st.checkbox("Show more results"):
                st.markdown(full_md, unsafe_allow_html=False)

        else:
            # No marker → just render whatever came back
            st.markdown(answer, unsafe_allow_html=False)

        # Finally, show sources/genes as before
        col1, col2 = st.columns([0.6, 0.4])
        if sources:
            with col1.expander("📄 Sources", expanded=True):
                show_sources(sources)
        if genes not in (None, [], ""):
            with col2.expander("🧬 Genes Mentioned", expanded=True):
                show_genes(genes)
