<div align="center">

# 📄 Research Paper Chatbot

**A RAG-powered chatbot that turns dense research papers into a conversation — with grounded, page-cited answers.**

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-Orchestration-1C3C3C)](https://www.langchain.com/)
[![Groq](https://img.shields.io/badge/LLM-Groq%20LLaMA%203.3%2070B-orange)](https://groq.com/)
[![License](https://img.shields.io/badge/License-MIT-green)](#license)

[**🔗 Live Demo**](https://research-paper-chatbot-genai.streamlit.app/) · [Features](#-features) · [Tech Stack](#%EF%B8%8F-tech-stack) · [Setup](#-getting-started-local-setup)

</div>

---

## 🖼️ Demo

<table>
<tr>
<td width="33%"><img src="readme_assets/screenshot-home.png" alt="Home screen"/><p align="center"><sub>Landing screen</sub></p></td>
<td width="33%"><img src="readme_assets/screenshot-upload.png" alt="Knowledge base built"/><p align="center"><sub>Knowledge base ready</sub></p></td>
<td width="33%"><img src="readme_assets/screenshot-answer.png" alt="Cited answer"/><p align="center"><sub>Grounded, cited answer</sub></p></td>
</tr>
</table>

**Try it live:** [research-paper-chatbot-genai.streamlit.app](https://research-paper-chatbot-genai.streamlit.app/)

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#%EF%B8%8F-tech-stack)
- [Architecture](#%EF%B8%8F-architecture)
- [Project Structure](#-project-structure)
- [Functional Requirements](#%EF%B8%8F-functional-requirements)
- [Key Technical Decisions](#-key-technical-decisions)
- [Getting Started](#-getting-started-local-setup)
- [Deployment](#%EF%B8%8F-deployment)
- [Future Improvements](#-future-improvements)
- [License](#license)

---

## 🚨 Problem Statement

Reading and extracting information from academic papers is slow and inefficient:

1. **Information overload** — A single paper can run 20–50 pages. Finding one specific result, method, or limitation means reading the whole thing.
2. **Manual cross-paper comparison** — Comparing methodology, contributions, or results between multiple papers requires juggling tabs, taking notes, and synthesizing by hand.
3. **Keyword search misses meaning** — Traditional search finds exact words, not the *concept* behind a question phrased differently than the paper's wording.

This project solves all three using **Retrieval-Augmented Generation (RAG)** — relevant sections of a paper are retrieved by semantic meaning, and the LLM is constrained to answer *only* from that retrieved content, eliminating hallucination and keeping every answer traceable back to an exact page.

---

## 📖 Overview

Upload one or more research papers and ask questions in plain English. The chatbot:
- Retrieves the most relevant sections of the paper(s) using semantic search
- Generates an answer using only that retrieved context
- Cites the exact source file and page number for every claim
- Remembers the conversation, so follow-up questions work naturally
- Tells you clearly when a topic isn't covered in the paper, instead of guessing

As shown in the demo above, asking *"Summarize this paper in a few sentences"* on a Nature paper about whale-fall ecosystems returns a structured, multi-paragraph answer — every fact tagged with its exact source page.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📚 Multi-PDF upload | Upload and query across multiple papers at once |
| 💬 Conversational memory | Follow-up questions work naturally — context carries over |
| 📌 Page-level citations | Every answer cites the exact source file and page number |
| 🎯 Confidence scoring | Each retrieved chunk shows a similarity confidence score |
| ⚡ Streaming responses | Answers stream token-by-token instead of loading all at once |
| 🚫 Out-of-scope detection | Clearly states when a topic isn't covered, instead of guessing |
| 🔍 Paper comparison mode | Structured side-by-side comparison across methodology, contributions, results |
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
