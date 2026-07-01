# UI
import streamlit as st

import config
import feedback
import llm
import rag
import vectorstore
from rag import retrieve

st.set_page_config(page_title="RAG Q&A", layout="centered")
st.title("RAG Question-Answering")
st.caption("Local • bge embeddings • Chroma • bge re-ranker • OCR • Mistral 7B")

# Settings in the Sidebar So the Pipeline Can Be Tuned per Query
with st.sidebar:
    st.header("Settings")
    # Restrict Retrieval to a Source Root / Folder. "All" = Unscoped (Needed for
    # Cross-Note Questions). Options Are the Real folders Discovered in the Index
    scope = st.selectbox("Search scope", ["All"] + vectorstore.list_folders(),
                         help="Limit retrieval to one source root or folder")
    scope = None if scope == "All" else scope
    # Post-Filter to Chunks Carrying a Tag (Frontmatter / Inline #tags). Combines With Scope
    tag = st.selectbox("Tag", ["Any"] + vectorstore.list_tags(),
                       help="Keep only chunks carrying this tag")
    tag = None if tag == "Any" else tag
    use_rewrite = st.toggle("Query rewriting", value=config.USE_REWRITE,
                            help="Slower (extra LLM calls); use when sources come back weak")
    use_rerank = st.toggle("Cross-encoder re-rank", value=config.USE_RERANK)
    use_hybrid = st.toggle("Hybrid (BM25 + vector)", value=config.USE_HYBRID,
                           help="Adds keyword search; helps code/acronym/exact-match corpora, "
                                "hurts clean prose. Off by default")
    k = st.slider("Chunks to LLM (top-k)", 1, 8, config.TOP_K)
    retrieve_n = st.slider(
        "Candidates before re-rank", k, 20, max(config.RETRIEVE_N, k),
        disabled=not use_rerank,
    )
    st.divider()
    st.caption(f"Index: `{config.CHROMA_DIR.name}`  •  Model: `{config.OLLAMA_MODEL}`")

question = st.text_input("Ask a question about your documents:")

if question:
    # Cache the Full Run per (Question + Settings). A Feedback-Button Click Reruns the
    # Whole Script; Without This It Would Re-Retrieve and Re-Generate the Answer Each Click
    key = (question, scope, tag, k, use_rerank, use_hybrid, use_rewrite, retrieve_n)
    fresh = st.session_state.get("key") != key

    if fresh:
        with st.spinner("Retrieving…"):
            r = retrieve(question, k=k, use_rerank=use_rerank, use_hybrid=use_hybrid,
                         use_rewrite=use_rewrite, retrieve_n=retrieve_n, scope=scope, tag=tag)
    else:
        r = st.session_state["r"]

    if r["rewrites"]:
        with st.expander(f"Also searched these {len(r['rewrites'])} rephrasings"):
            for rw in r["rewrites"]:
                st.markdown(f"- {rw}")

    st.markdown("### Answer")
    if r["abstain"]:
        # Retrieval Too Weak - Refuse Instead of Answering From Loosely-Related Chunks
        answer_text, verified, vreason = llm.REFUSAL, None, ""
        st.info(answer_text)
        st.caption(f"abstained — rerank {r['top_score']} / similarity {r['top_sim']} both below threshold")
    elif fresh:
        answer_text = st.write_stream(llm.generate_stream(question, r["blocks"]))  # Streams + Returns Full Text
        verified, vreason = rag.verify_answer(r, answer_text)  # Gray-Zone Grounding Check
    else:
        answer_text, verified, vreason = (st.session_state[x] for x in ("answer", "verified", "vreason"))
        st.write(answer_text)

    if verified is False:  # Post-Generation Check Couldn't Confirm Grounding - Flag, Don't Delete
        st.warning(f"⚠️ May not be fully grounded: {vreason}")

    if fresh:
        st.session_state.update(key=key, r=r, answer=answer_text, verified=verified, vreason=vreason)
        feedback.log_query(question, {**r, "verified": verified})

    st.caption(f"retrieval {r['retrieve_s']}s • {r['retrieval']}"
               f"{' • re-ranked' if r['reranked'] else ''}")

    st.markdown("### Closest matches" if r["abstain"] else "### Sources")
    for s in r["sources"]:
        score = (
            f"rerank {s['rerank_score']}  ·  similarity {s['score']}"
            if s["rerank_score"] is not None
            else f"similarity {s['score']}"
        )
        with st.expander(f"{s['source']}  —  {score}"):
            st.write(s["text"])

    # Thumbs Feedback -> logs/feedback.jsonl. Buttons Rerun the Script; the Cache Above
    # Keeps That From Re-Generating the Answer
    st.divider()
    c1, c2, _ = st.columns([1, 1, 8])
    if c1.button("👍", help="Helpful"):
        feedback.log_feedback(question, answer_text, "up")
        st.caption("feedback logged — thanks")
    if c2.button("👎", help="Not helpful"):
        feedback.log_feedback(question, answer_text, "down")
        st.caption("feedback logged — thanks")
