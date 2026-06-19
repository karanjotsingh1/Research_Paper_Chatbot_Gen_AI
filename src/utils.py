SUGGESTED_QUESTIONS = [
    "Summarize this paper in a few sentences",
    "What problem does this paper solve?",
    "What methodology or approach was proposed?",
    "What are the key contributions?",
    "What are the limitations and future directions?",
    "Who are the authors and when was it published?",
    "Explain this paper in simple language",
]


def export_chat_to_text(chat_history: list) -> str:
    """Converts the chat history into a plain-text downloadable transcript."""
    lines = ["RESEARCH PAPER CHATBOT — CONVERSATION TRANSCRIPT", "=" * 50, ""]
    for msg in chat_history:
        role = "You" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
        lines.append("")
    return "\n".join(lines)


def format_sources(source_documents: list) -> list:
    """Deduplicates and formats retrieved chunks for the citations expander."""
    seen = set()
    formatted = []
    for doc in source_documents:
        source = doc.metadata.get("source_file", "Unknown")
        page = doc.metadata.get("display_page", "N/A")
        key = (source, page)
        if key in seen:
            continue
        seen.add(key)
        snippet = doc.page_content[:200] + ("..." if len(doc.page_content) > 200 else "")
        formatted.append({"source": source, "page": page, "snippet": snippet})
    return sorted(formatted, key=lambda x: (x["source"], x["page"]))
