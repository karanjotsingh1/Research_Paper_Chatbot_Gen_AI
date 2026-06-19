from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP


def split_documents(documents: list) -> list:
    """
    Splits documents into overlapping chunks.

    - chunk_size=1000: large enough to hold a complete idea (a methodology
      paragraph, a results section) without losing meaning.
    - chunk_overlap=200: prevents key sentences from being split across
      two chunks and becoming un-retrievable.
    - Separator order tries paragraph → sentence → word breaks, so we
      never cut mid-sentence when avoidable.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    # Tag each chunk with an index for traceability
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    return chunks
