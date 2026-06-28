"""Streamlit demo UI for the RAG system. Run: streamlit run app.py"""
import streamlit as st

import config
import llm
from rag import retrieve

st.set_page_config(page_title="RAG Q&A", page_icon="📚", layout="centered")
st.title("📚 RAG Question-Answering")
st.caption("Local • bge embeddings • Chroma • bge re-ranker • OCR • Mistral 7B")

# Settings live in the sidebar so the pipeline can be tuned per query
with st.sidebar:
    st.header("Settings")
    use_rewrite = st.toggle("Query rewriting", value=config.USE_REWRITE,
                            help="Slower (extra LLM calls); use when sources come back weak")
    use_rerank = st.toggle("Cross-encoder re-rank", value=config.USE_RERANK)
    k = st.slider("Chunks to LLM (top-k)", 1, 8, config.TOP_K)
    retrieve_n = st.slider(
        "Candidates before re-rank", k, 20, max(config.RETRIEVE_N, k),
        disabled=not use_rerank,
    )
    st.divider()
    st.caption(f"Index: `{config.CHROMA_DIR.name}`  •  Model: `{config.OLLAMA_MODEL}`")

question = st.text_input("Ask a question about your documents:")

if question:
    with st.spinner("Retrieving…"):
        r = retrieve(question, k=k, use_rerank=use_rerank,
                     use_rewrite=use_rewrite, retrieve_n=retrieve_n)

    if r["rewrites"]:
        with st.expander(f"🔁 Also searched these {len(r['rewrites'])} rephrasings"):
            for rw in r["rewrites"]:
                st.markdown(f"- {rw}")

    st.markdown("### Answer")
    st.write_stream(llm.generate_stream(question, r["blocks"]))  # tokens as generated

    st.caption(f"⏱ retrieval {r['retrieve_s']}s • {r['retrieval']}"
               f"{' • re-ranked' if r['reranked'] else ''}")

    st.markdown("### Sources")
    for s in r["sources"]:
        score = (
            f"rerank {s['rerank_score']}  ·  similarity {s['score']}"
            if s["rerank_score"] is not None
            else f"similarity {s['score']}"
        )
        with st.expander(f"📄 {s['source']}  —  {score}"):
            st.write(s["text"])
