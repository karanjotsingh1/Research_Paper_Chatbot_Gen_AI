import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ── LLM Settings ──────────────────────────────────────────────────────────
LLM_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.2

# ── Embedding Settings ───────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── Chunking Settings ─────────────────────────────────────────────────────
CHUNK_SIZE = 1200      # slightly larger chunks = more context per retrieved piece
CHUNK_OVERLAP = 300    # bigger overlap = less chance of a key sentence being cut off

# ── Retrieval Settings ───────────────────────────────────────────────────
RETRIEVAL_TOP_K = 8    # more chunks = more context = more complete answers

# ── Paths ─────────────────────────────────────────────────────────────────
UPLOAD_DIR = "data/uploads"
VECTORSTORE_DIR = "vectorstore"

for d in [UPLOAD_DIR, VECTORSTORE_DIR]:
    os.makedirs(d, exist_ok=True)
