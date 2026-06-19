from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.vectorstores import FAISS
from src.prompts import answer_prompt, contextualize_prompt, comparison_prompt
from config import GROQ_API_KEY, LLM_MODEL, LLM_TEMPERATURE, RETRIEVAL_TOP_K


def get_llm() -> ChatGroq:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found. Add it to your .env file.")
    return ChatGroq(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        groq_api_key=GROQ_API_KEY,
        max_tokens=4096,   # default is ~1024 — this is why answers were short
    )

def format_docs(docs: list) -> str:
    """Formats retrieved chunks into a readable, citation-tagged context block."""
    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source_file", "Unknown")
        page = doc.metadata.get("display_page", "N/A")
        parts.append(f"[Chunk {i} | {source} | Page {page}]\n{doc.page_content}")
    return "\n\n".join(parts)


def build_rag_chain(vector_store: FAISS):
    """
    Builds the full conversational RAG pipeline:

    1. If there's chat history, rewrite the question to be standalone.
    2. Retrieve the top-K most relevant chunks from FAISS using MMR
       (Maximum Marginal Relevance) — ensures diverse, non-redundant results.
    3. Format chunks into a context block with source citations.
    4. Feed context + question + history into the answer prompt → LLM → stream.

    Returns (retrieve_fn, streaming_chain) so the UI can display sources
    separately from the streamed answer.
    """
    llm = get_llm()
    parser = StrOutputParser()

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": RETRIEVAL_TOP_K, "fetch_k": RETRIEVAL_TOP_K * 2},
    )

    contextualize_chain = contextualize_prompt | llm | parser

    def get_question(input_dict: dict) -> str:
        """Rewrites follow-up questions into standalone ones."""
        if input_dict.get("chat_history"):
            return contextualize_chain.invoke(input_dict)
        return input_dict["question"]

    retrieve_docs = RunnableLambda(get_question) | retriever

    streaming_chain = (
        RunnablePassthrough.assign(context=lambda x: format_docs(x["source_documents"]))
        | answer_prompt
        | llm
        | parser
    )

    return retrieve_docs, streaming_chain


def build_comparison_chain():
    """Dedicated chain for the Compare Papers feature."""
    llm = get_llm()
    return comparison_prompt | llm | StrOutputParser()
