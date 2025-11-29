import sys
from pathlib import Path
import os

# ------------------------------------------------------------------------
# FIX IMPORT PATH (MUST BE VERY TOP)
# ------------------------------------------------------------------------
# Ensure project root is on sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

# ------------------------------------------------------------------------
# External imports
# ------------------------------------------------------------------------
import streamlit as st
import chromadb
from chromadb.config import Settings

# ------------------------------------------------------------------------
# MUST BE FIRST STREAMLIT COMMAND
# ------------------------------------------------------------------------
st.set_page_config(
    page_title="AeroSense RAG – UAV Troubleshooting",
    layout="wide",
)

# ------------------------------------------------------------------------
# INTERNAL IMPORTS (must be AFTER sys.path + AFTER set_page_config)
# ------------------------------------------------------------------------
from rag_pipeline.config import paths, retrieval_cfg, models
from rag_pipeline.retrieval import retrieve_uav_docs
from rag_pipeline.llm_inference import generate_answer

# ------------------------------------------------------------------------
# DEBUG BLOCK
# ------------------------------------------------------------------------
st.write("DEBUG — Streamlit CWD:", os.getcwd())

client = chromadb.PersistentClient(
    path=str(paths.vector_db_dir),
    settings=Settings(allow_reset=True)
)

st.write("DEBUG — Collections loaded by Streamlit:", client.list_collections())

# ------------------------------------------------------------------------
# UI
# ------------------------------------------------------------------------
st.title("🛠️ AeroSense RAG – UAV Troubleshooting Assistant")
st.caption("Engineering manuals + telemetry logs · Local embeddings · Local LLM via Ollama")

with st.sidebar:
    st.header("Settings")

    top_k = st.slider("Top-K results (global)", 3, 15, retrieval_cfg.top_k)
    manual_weight = st.slider("Manual weight", 0.0, 1.0, retrieval_cfg.manual_weight, 0.05)
    telemetry_weight = 1.0 - manual_weight

    temperature = st.slider("LLM temperature", 0.0, 1.0, 0.2, 0.05)

    st.markdown(f"**Embedding model:** `{models.embedding_model_name}`")
    st.markdown(f"**Ollama model:** `{models.ollama_model}`")

st.markdown("### Describe the UAV issue")

query = st.text_input(
    "Example: ESC overheating during high-altitude climb with intermittent GPS dropout",
    key="user_query",
)

col_ctx, col_ans = st.columns([2, 3])

# ------------------------------------------------------------------------
# MAIN ACTION BUTTON
# ------------------------------------------------------------------------
if st.button("Diagnose", type="primary") and query.strip():

    # Dynamically update config
    retrieval_cfg.top_k = top_k
    retrieval_cfg.manual_weight = manual_weight
    retrieval_cfg.log_weight = telemetry_weight

    with st.spinner("Retrieving relevant manual sections and telemetry segments..."):
        retrieved = retrieve_uav_docs(query)

    # -----------------------------
    # CONTEXT PANEL
    # -----------------------------
    with col_ctx:
        st.subheader("Retrieved Context")

        if not retrieved:
            st.warning("No documents retrieved. Check if your index is built and data exists.")
        else:
            for i, doc in enumerate(retrieved, start=1):
                with st.expander(f"#{i} [{doc.source_type.upper()}] score={doc.score:.3f}"):
                    src = doc.metadata.get("source", "")
                    ts = doc.metadata.get("timestamp", "")

                    if src:
                        st.markdown(f"**Source:** `{src}`")
                    if ts:
                        st.markdown(f"**Timestamp:** `{ts}`")

                    st.markdown("---")
                    snippet = doc.text[:1200] + ("..." if len(doc.text) > 1200 else "")
                    st.write(snippet)

    # -----------------------------
    # ANSWER PANEL
    # -----------------------------
    with col_ans:
        st.subheader("Troubleshooting Suggestion")

        if not retrieved:
            st.info("No context available. Try a different query or rebuild your index.")
        else:
            with st.spinner("Calling local LLM via Ollama..."):
                answer = generate_answer(query, retrieved, temperature=temperature)
            st.markdown(answer)

        # Debug metadata summary
        st.markdown("---")
        st.markdown("#### Debug info")
        if retrieved:
            for i, doc in enumerate(retrieved, start=1):
                st.markdown(
                    f"- #{i} | `{doc.source_type}` | score={doc.score:.3f} "
                    f"| source=`{doc.metadata.get('source','')}`"
                )
