import os
import hashlib
from langchain_community.document_loaders import PyPDFLoader


def get_combined_hash(file_paths: list) -> str:
    """
    Creates a unique hash for the exact set of uploaded PDFs.
    Used as a cache key — same papers = reuse saved FAISS index.
    """
    hasher = hashlib.md5()
    for path in sorted(file_paths):
        with open(path, "rb") as f:
            hasher.update(f.read())
    return hasher.hexdigest()[:12]


def load_pdfs(file_paths: list) -> dict:
    """
    Loads one or more PDFs. Tags every page with its source filename
    and a human-readable page number for citations.
    Errors on individual files are collected without crashing the batch.
    """
    all_documents = []
    filenames = []
    errors = []

    for path in file_paths:
        filename = os.path.basename(path)
        try:
            loader = PyPDFLoader(path)
            docs = loader.load()

            if not docs:
                errors.append(f"'{filename}' appears empty or unreadable.")
                continue

            for doc in docs:
                # Normalize whitespace
                doc.page_content = " ".join(doc.page_content.split())
                doc.metadata["source_file"] = filename
                # PyPDFLoader uses 0-indexed pages — make it human-readable
                doc.metadata["display_page"] = doc.metadata.get("page", 0) + 1

            all_documents.extend(docs)
            filenames.append(filename)

        except Exception as e:
            errors.append(f"Could not read '{filename}': {str(e)}")

    return {"documents": all_documents, "filenames": filenames, "errors": errors}
