import streamlit as st
import os
from langchain_core.messages import HumanMessage, AIMessage

from config import GROQ_API_KEY, UPLOAD_DIR
from src.document_loader import load_pdfs, get_combined_hash
from src.text_splitter import split_documents
from src.vector_store import build_or_load_vector_store
from src.rag_chain import build_rag_chain, build_comparison_chain, format_docs
from src.utils import SUGGESTED_QUESTIONS, export_chat_to_text, format_sources

st.set_page_config(page_title="Research Paper Chatbot", page_icon="📄", layout="wide")

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
if "processed" not in st.session_state:
    st.session_state.processed = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []       # list of {"role", "content", "sources"}
if "retrieve_docs" not in st.session_state:
    st.session_state.retrieve_docs = None
if "streaming_chain" not in st.session_state:
    st.session_state.streaming_chain = None
if "filenames" not in st.session_state:
    st.session_state.filenames = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.title("📄 Research Paper Chatbot")
st.sidebar.caption("RAG-powered Q&A over your research papers")

if not GROQ_API_KEY:
    st.sidebar.error("⚠️ GROQ_API_KEY not set. Add it to your .env file.")

st.sidebar.markdown("---")
st.sidebar.subheader("1. Upload Papers")

uploaded_files = st.sidebar.file_uploader(
    "Upload one or more PDFs",
    type=["pdf"],
    accept_multiple_files=True,
)

build_btn = st.sidebar.button("🔨 Build Knowledge Base", type="primary", use_container_width=True)

if uploaded_files:
    st.sidebar.markdown("**Uploaded:**")
    for f in uploaded_files:
        st.sidebar.caption(f"📄 {f.name}")

# ── Build pipeline on button click ───────────────────────────────────────────
if build_btn:
    if not uploaded_files:
        st.sidebar.warning("Please upload at least one PDF first.")
        st.stop()

    # Save uploaded files to disk
    file_paths = []
    for f in uploaded_files:
        path = os.path.join(UPLOAD_DIR, f.name)
        with open(path, "wb") as out:
            out.write(f.read())
        file_paths.append(path)

    cache_key = get_combined_hash(file_paths)

    with st.spinner("📖 Reading PDFs..."):
        result = load_pdfs(file_paths)

    for err in result["errors"]:
        st.sidebar.error(err)

    if not result["documents"]:
        st.sidebar.error("No readable content found. Check that the PDFs are not scanned images.")
        st.stop()

    with st.spinner("✂️ Splitting into chunks..."):
        chunks = split_documents(result["documents"])

    with st.spinner("🔍 Building vector index (cached for repeat uploads)..."):
        vector_store = build_or_load_vector_store(chunks, cache_key)

    with st.spinner("⚙️ Setting up RAG pipeline..."):
        retrieve_docs, streaming_chain = build_rag_chain(vector_store)

    # Save to session state
    st.session_state.processed = True
    st.session_state.retrieve_docs = retrieve_docs
    st.session_state.streaming_chain = streaming_chain
    st.session_state.vector_store = vector_store
    st.session_state.filenames = result["filenames"]
    st.session_state.chat_history = []

    st.sidebar.success(
        f"✅ Ready! {len(chunks)} chunks indexed from {len(result['filenames'])} paper(s)."
    )

st.sidebar.markdown("---")

# ── Utility buttons ───────────────────────────────────────────────────────────
if st.sidebar.button("🗑️ Clear Chat", use_container_width=True):
    st.session_state.chat_history = []
    st.rerun()

if st.session_state.chat_history:
    transcript = export_chat_to_text(st.session_state.chat_history)
    st.sidebar.download_button(
        "⬇️ Download Chat",
        data=transcript,
        file_name="chat_transcript.txt",
        mime="text/plain",
        use_container_width=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════════════════
st.title("📄 Research Paper Chatbot")
st.caption("Ask questions about your papers — grounded answers with page-level citations, powered by RAG + Groq.")

if not st.session_state.processed:
    st.info("👈 Upload one or more research papers and click **Build Knowledge Base** to begin.")

    with st.expander("ℹ️ How it works"):
        st.markdown("""
        1. **Load** — PDFs are parsed page-by-page, preserving page numbers and filenames for citations.
        2. **Chunk** — Pages are split into overlapping 1000-character chunks so no idea gets cut off.
        3. **Embed** — Chunks are converted to vectors using `all-MiniLM-L6-v2` (a sentence-transformer model).
        4. **Index** — Vectors are stored in a FAISS index, saved to disk so repeat uploads are instant.
        5. **Retrieve** — On each question, MMR (Maximum Marginal Relevance) retrieval fetches the most relevant *and* diverse chunks.
        6. **Generate** — Retrieved context + your question + chat history are sent to Groq's LLaMA 70B model, streamed back in real time.
        """)
    st.stop()

# Active knowledge base banner
st.success(f"📚 Active: {', '.join(st.session_state.filenames)}")

# ── Quick question buttons ────────────────────────────────────────────────────
st.markdown("**Quick questions:**")
cols = st.columns(4)
clicked = None
for i, q in enumerate(SUGGESTED_QUESTIONS[:4]):
    if cols[i % 4].button(q, key=f"q_{i}", use_container_width=True):
        clicked = q

cols2 = st.columns(3)
for i, q in enumerate(SUGGESTED_QUESTIONS[4:]):
    if cols2[i % 3].button(q, key=f"q_extra_{i}", use_container_width=True):
        clicked = q

# ── Compare papers (only shown with 2+ papers) ───────────────────────────────
if len(st.session_state.filenames) >= 2:
    with st.expander("🔍 Compare uploaded papers"):
        compare_q = st.text_input(
            "What would you like to compare?",
            value="Compare these papers in terms of methodology and contributions",
            key="compare_input",
        )
        if st.button("Compare Papers", key="compare_btn"):
            with st.spinner("Retrieving context from all papers..."):
                docs = st.session_state.retrieve_docs.invoke({
                    "question": compare_q,
                    "chat_history": [],
                })
                context = format_docs(docs)

            with st.spinner("Generating comparison..."):
                chain = build_comparison_chain()
                result = chain.invoke({"question": compare_q, "context": context})

            st.markdown("### Comparison")
            st.markdown(result)
            with st.expander("Sources used"):
                for s in format_sources(docs):
                    st.caption(f"📄 {s['source']} — Page {s['page']}: {s['snippet']}")

st.markdown("---")

# ── Chat history display ──────────────────────────────────────────────────────
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📌 Source pages"):
                for s in msg["sources"]:
                    st.caption(f"📄 {s['source']} — Page {s['page']}: {s['snippet']}")

# ── Chat input ────────────────────────────────────────────────────────────────
user_question = st.chat_input("Ask a question about the paper(s)...") or clicked

if user_question:
    # Add user message
    st.session_state.chat_history.append({"role": "user", "content": user_question, "sources": None})
    with st.chat_message("user"):
        st.markdown(user_question)

    # Build LangChain message format for history (exclude the current question)
    lc_history = []
    for m in st.session_state.chat_history[:-1]:
        if m["role"] == "user":
            lc_history.append(HumanMessage(content=m["content"]))
        else:
            lc_history.append(AIMessage(content=m["content"]))

    with st.chat_message("assistant"):
        # Step 1: retrieve relevant chunks
        with st.spinner("Searching paper(s)..."):
            docs = st.session_state.retrieve_docs.invoke({
                "question": user_question,
                "chat_history": lc_history,
            })

        # Step 2: stream the answer
        def token_stream():
            for chunk in st.session_state.streaming_chain.stream({
                "question": user_question,
                "chat_history": lc_history,
                "source_documents": docs,
            }):
                yield chunk

        answer = st.write_stream(token_stream())

        # Step 3: show sources
        sources = format_sources(docs)
        if sources:
            with st.expander("📌 Source pages"):
                for s in sources:
                    st.caption(f"📄 {s['source']} — Page {s['page']}: {s['snippet']}")

    # Save assistant reply
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
