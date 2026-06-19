# 📄 Research Paper Chatbot

A Retrieval-Augmented Generation (RAG) powered chatbot that lets you upload research papers and have a grounded, citation-backed conversation with them — instead of reading 30 pages to find one answer.

**🔗 Live Demo:** [research-paper-chatbot-genai.streamlit.app](https://research-paper-chatbot-genai.streamlit.app/)

---

## 📖 Overview

Reading academic papers is slow. Finding one specific result, method, or limitation often means skimming an entire 20–50 page PDF. This project solves that by combining semantic search with a large language model — every answer is generated **only** from the actual content of the uploaded paper(s), with the exact page number cited, so there's no hallucination and no guesswork.

Upload one paper and ask questions about it, or upload several and ask the chatbot to compare them side-by-side.

---

## 🚨 Problem Statement

Working with research papers presents three recurring problems:

1. **Information overload** — locating a specific detail requires reading the whole document.
2. **Manual comparison across papers** — comparing methodology, results, or contributions between multiple papers means juggling tabs and taking notes by hand.
3. **Context loss** — keyword search finds words, not meaning, and misses information that's phrased differently than the search query.

This project solves all three using RAG: relevant chunks of the paper are retrieved by *meaning*, not just keywords, and the LLM is constrained to answer only from that retrieved context — giving accurate, traceable answers.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📚 Multi-PDF upload | Upload and query across multiple papers at once |
| 💬 Conversational memory | Ask follow-up questions naturally — context carries over |
| 📌 Page-level citations | Every answer cites the exact source file and page number |
| 🎯 Confidence scoring | Each retrieved chunk shows a similarity confidence score |
| ⚡ Streaming responses | Answers stream token-by-token instead of loading all at once |
| 🚫 Out-of-scope detection | Clearly states when a topic isn't covered in the paper, instead of guessing |
| 🔍 Paper comparison mode | Structured side-by-side comparison of methodology, contributions, and results |
| 💾 Persistent vector index | Re-uploading the same paper skips re-processing (local mode) |
| ⬇️ Exportable chat history | Download the full conversation as a `.txt` transcript |
| 🖱️ Quick-question buttons | One-click access to common queries like "summarize this paper" |

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **UI** | Streamlit | Interactive Python web app |
| **LLM** | Groq API — LLaMA 3.3 70B | Fast, high-quality answer generation |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | Converts text chunks into semantic vectors |
| **Vector Store** | FAISS | Stores and searches embedding vectors |
| **Orchestration** | LangChain | Chains, prompts, retrievers, conversational memory |
| **PDF Parsing** | PyPDFLoader | Extracts text with page-level metadata |
| **Retrieval Strategy** | MMR (Maximum Marginal Relevance) | Retrieves relevant *and* diverse chunks |
| **Text Splitting** | RecursiveCharacterTextSplitter | Smart, overlap-aware chunking |
| **Config** | python-dotenv | Environment variable management |

---

## 🏗️ Architecture

```
PDF Upload
   │
   ▼
PyPDFLoader ── extracts text page-by-page, tags source filename + page number
   │
   ▼
RecursiveCharacterTextSplitter ── splits into 1200-character overlapping chunks
   │
   ▼
HuggingFace Embeddings ── converts each chunk into a dense vector
   │
   ▼
FAISS Vector Store ── indexes all vectors for fast similarity search
   │
   ▼
User asks a question
   │
   ├── If follow-up question → LLM rewrites it into a standalone question
   │
   ▼
MMR Retriever ── fetches top-8 most relevant + diverse chunks, with confidence scores
   │
   ▼
Context Assembly ── chunks formatted with [source file | page number] tags
   │
   ▼
LLaMA 3.3 70B (via Groq) ── generates a grounded, cited answer
   │
   ▼
Streamed to UI ── token-by-token, with source pages + confidence shown below
```

---

## 📂 Project Structure

```
research_paper_chatbot/
├── app.py                  # Streamlit UI — main entry point
├── config.py               # Centralized settings (models, chunk sizes, paths)
├── requirements.txt        # Python dependencies
├── packages.txt            # System-level dependencies (for cloud deployment)
├── .gitignore
├── .env                    # API keys (not committed)
├── src/
│   ├── document_loader.py  # PDF loading, page tagging, hash-based caching
│   ├── text_splitter.py    # Chunking logic
│   ├── vector_store.py     # FAISS index build/load
│   ├── rag_chain.py        # Retrieval + LLM chains, confidence scoring
│   ├── prompts.py          # All system prompts and templates
│   └── utils.py            # Helper functions (formatting, export)
├── data/uploads/            # Saved PDFs (auto-created, gitignored)
└── vectorstore/             # Persisted FAISS indexes (auto-created, gitignored)
```

---

## ⚙️ Functional Requirements

| # | Requirement |
|---|---|
| FR1 | Accept one or more PDF uploads |
| FR2 | Extract text with page-level metadata |
| FR3 | Split text into overlapping, context-preserving chunks |
| FR4 | Embed chunks using a sentence-transformer model |
| FR5 | Store and retrieve embeddings via a FAISS vector index |
| FR6 | Generate answers strictly from retrieved context (no hallucination) |
| FR7 | Cite the source file and page number for every factual claim |
| FR8 | Support multi-turn conversations with memory |
| FR9 | Rewrite ambiguous follow-up questions into standalone queries |
| FR10 | Detect and clearly flag questions outside the paper's scope |
| FR11 | Provide a structured comparison mode for multiple papers |
| FR12 | Stream the LLM's response token-by-token |
| FR13 | Display a confidence/similarity score for each retrieved chunk |
| FR14 | Handle unreadable or corrupted PDFs without crashing |
| FR15 | Export the full conversation as a downloadable text file |

---

## 🧠 Key Technical Decisions

**Why RAG instead of feeding the whole PDF to the LLM?**
LLMs have a limited context window. A 50-page paper can exceed it. RAG retrieves only the most relevant chunks — usually 8 — keeping token usage low while preserving answer accuracy.

**Why MMR (Maximum Marginal Relevance) instead of plain similarity search?**
Plain similarity search often returns several near-duplicate chunks that all say the same thing. MMR balances relevance with diversity, so the retrieved context covers different angles of the answer instead of repeating one point eight times.

**Why rewrite follow-up questions before retrieval?**
A question like *"What about the limitations?"* is meaningless to a vector search on its own. The system uses the conversation history to rewrite it into a standalone question — *"What are the limitations of the proposed methodology?"* — before retrieval happens.

**Why chunk with overlap?**
Key sentences often fall at the boundary between two chunks. A 300-character overlap ensures that boundary content appears in both neighboring chunks, so nothing falls through the cracks.

**Why show a confidence score per chunk?**
Cosine similarity between the question vector and each retrieved chunk gives a quantitative sense of how well-supported an answer is — useful for users deciding how much to trust a given response.

---

## 🚀 Getting Started (Local Setup)

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/research-paper-chatbot.git
cd research-paper-chatbot

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Groq API key
echo "GROQ_API_KEY=your-key-here" > .env

# 5. Run the app
streamlit run app.py
```

Get a free Groq API key at [console.groq.com](https://console.groq.com)

---

## ☁️ Deployment

This app is deployed on **Streamlit Community Cloud**. To deploy your own copy:

1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. Click **New app**, select this repo, branch `main`, and main file `app.py`
4. Under **Advanced settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "your-key-here"
   ```
5. Click **Deploy**

---

## 🔮 Future Improvements

- OCR support for scanned/image-based PDFs using Tesseract
- A dedicated reranker model to re-score retrieved chunks before generation
- Switch to a larger embedding model (`all-mpnet-base-v2`) for improved retrieval accuracy
- Persistent cloud-based vector storage (e.g., Pinecone or Weaviate) instead of local FAISS
- User authentication and per-user document libraries

---

## 📜 License

This project is open-source and available for educational and personal use.
