📄 Research Paper Chatbot
A RAG-powered chatbot that turns dense research papers into a conversation — with grounded, page-cited answers.

🔗 Live Demo · Features · Tech Stack · Setup
</div>

🖼️ Demo
<table>
<tr>
<td width="33%"><img src="readme_assets/screenshot-home.png" alt="Home screen"/><p align="center"><sub>Landing screen</sub></p></td>
<td width="33%"><img src="readme_assets/screenshot-upload.png" alt="Knowledge base built"/><p align="center"><sub>Knowledge base ready</sub></p></td>
<td width="33%"><img src="readme_assets/screenshot-answer.png" alt="Cited answer"/><p align="center"><sub>Grounded, cited answer</sub></p></td>
</tr>
</table>
Try it live: research-paper-chatbot-genai.streamlit.app

📋 Table of Contents

Problem Statement
Overview
Features
Tech Stack
Architecture
Project Structure
Functional Requirements
Key Technical Decisions
Getting Started
Deployment
Future Improvements
License


🚨 Problem Statement
Reading and extracting information from academic papers is slow and inefficient:

Information overload — A single paper can run 20–50 pages. Finding one specific result, method, or limitation means reading the whole thing.
Manual cross-paper comparison — Comparing methodology, contributions, or results between multiple papers requires juggling tabs, taking notes, and synthesizing by hand.
Keyword search misses meaning — Traditional search finds exact words, not the concept behind a question phrased differently than the paper's wording.

This project solves all three using Retrieval-Augmented Generation (RAG) — relevant sections of a paper are retrieved by semantic meaning, and the LLM is constrained to answer only from that retrieved content, eliminating hallucination and keeping every answer traceable back to an exact page.

📖 Overview
Upload one or more research papers and ask questions in plain English. The chatbot:

Retrieves the most relevant sections of the paper(s) using semantic search
Generates an answer using only that retrieved context
Cites the exact source file and page number for every claim
Remembers the conversation, so follow-up questions work naturally
Tells you clearly when a topic isn't covered in the paper, instead of guessing

As shown in the demo above, asking "Summarize this paper in a few sentences" on a Nature paper about whale-fall ecosystems returns a structured, multi-paragraph answer — every fact tagged with its exact source page.

✨ Features
FeatureDescription📚 Multi-PDF uploadUpload and query across multiple papers at once💬 Conversational memoryFollow-up questions work naturally — context carries over📌 Page-level citationsEvery answer cites the exact source file and page number🎯 Confidence scoringEach retrieved chunk shows a similarity confidence score⚡ Streaming responsesAnswers stream token-by-token instead of loading all at once🚫 Out-of-scope detectionClearly states when a topic isn't covered, instead of guessing🔍 Paper comparison modeStructured side-by-side comparison across methodology, contributions, results💾 Persistent vector indexRe-uploading the same paper skips re-processing (local mode)⬇️ Exportable chat historyDownload the full conversation as a .txt transcript🖱️ Quick-question buttonsOne-click access to common queries like "summarize this paper"

🛠️ Tech Stack
LayerTechnologyPurposeUIStreamlitInteractive Python web appLLMGroq API — LLaMA 3.3 70BFast, high-quality answer generationEmbeddingssentence-transformers/all-MiniLM-L6-v2Converts text chunks into semantic vectorsVector StoreFAISSStores and searches embedding vectorsOrchestrationLangChainChains, prompts, retrievers, conversational memoryPDF ParsingPyPDFLoaderExtracts text with page-level metadataRetrieval StrategyMMR (Maximum Marginal Relevance)Retrieves relevant and diverse chunksText SplittingRecursiveCharacterTextSplitterSmart, overlap-aware chunkingConfigpython-dotenvEnvironment variable management

🏗️ Architecture
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

📂 Project Structure
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

⚙️ Functional Requirements
#RequirementFR1Accept one or more PDF uploadsFR2Extract text with page-level metadataFR3Split text into overlapping, context-preserving chunksFR4Embed chunks using a sentence-transformer modelFR5Store and retrieve embeddings via a FAISS vector indexFR6Generate answers strictly from retrieved context (no hallucination)FR7Cite the source file and page number for every factual claimFR8Support multi-turn conversations with memoryFR9Rewrite ambiguous follow-up questions into standalone queriesFR10Detect and clearly flag questions outside the paper's scopeFR11Provide a structured comparison mode for multiple papersFR12Stream the LLM's response token-by-tokenFR13Display a confidence/similarity score for each retrieved chunkFR14Handle unreadable or corrupted PDFs without crashingFR15Export the full conversation as a downloadable text file

🧠 Key Technical Decisions
Why RAG instead of feeding the whole PDF to the LLM?

LLMs have a limited context window. A 50-page paper can exceed it. RAG retrieves only the most relevant chunks — usually 8 — keeping token usage low while preserving answer accuracy.
Why MMR (Maximum Marginal Relevance) instead of plain similarity search?

Plain similarity search often returns several near-duplicate chunks that all say the same thing. MMR balances relevance with diversity, so retrieved context covers different angles of the answer instead of repeating one point eight times.
Why rewrite follow-up questions before retrieval?

A question like "What about the limitations?" is ambiguous to a vector search on its own. The system uses conversation history to rewrite it into a standalone question — "What are the limitations of the proposed methodology?" — before retrieval happens.
Why chunk with overlap?

Key sentences often fall at the boundary between two chunks. A 300-character overlap ensures boundary content appears in both neighboring chunks, so nothing falls through the cracks.
Why show a confidence score per chunk?

Cosine similarity between the question vector and each retrieved chunk gives a quantitative sense of how well-supported an answer is — useful for users deciding how much to trust a given response.

🚀 Getting Started (Local Setup)
bash# 1. Clone the repository
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
Get a free Groq API key at console.groq.com

☁️ Deployment
This app is deployed on Streamlit Community Cloud:
🔗 research-paper-chatbot-genai.streamlit.app
To deploy your own copy:

Push this repository to GitHub
Go to share.streamlit.io and sign in with GitHub
Click New app, select this repo, branch main, and main file app.py
Under Advanced settings → Secrets, add:

toml   GROQ_API_KEY = "your-key-here"

Click Deploy


🔮 Future Improvements

OCR support for scanned/image-based PDFs using Tesseract
A dedicated reranker model to re-score retrieved chunks before generation
Switch to a larger embedding model (all-mpnet-base-v2) for improved retrieval accuracy
Persistent cloud-based vector storage (e.g., Pinecone or Weaviate) instead of local FAISS
User authentication and per-user document libraries


License
This project is open-source and available for educational and personal use under the MIT License.
