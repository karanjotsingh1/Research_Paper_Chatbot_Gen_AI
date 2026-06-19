# 📄 Research Paper Chatbot using RAG, LangChain & Groq

An industry-level **Retrieval-Augmented Generation (RAG)** application that enables users to upload one or more research papers and interact with them using natural language questions. The system leverages **LangChain, FAISS, Hybrid Retrieval, MultiQueryRetriever, and Groq LLMs** to provide grounded, context-aware answers with page-level citations.

---

## 🚀 Project Overview

Research papers are often lengthy, highly technical, and time-consuming to understand. This project addresses that problem by providing an AI-powered assistant capable of:

* Summarizing research papers
* Answering questions from uploaded PDFs
* Explaining technical concepts in simple language
* Comparing multiple research papers
* Maintaining conversational context
* Providing source citations and page references

The application is designed using modern **Generative AI and RAG architecture patterns** commonly used in production AI systems.

---

## 🎯 Problem Statement

Researchers and students spend significant time:

* Reading long academic papers
* Searching for specific information
* Understanding technical terminology
* Comparing multiple research works
* Extracting key contributions and methodologies

The goal of this project is to build an intelligent assistant that can understand uploaded research papers and answer questions accurately by retrieving relevant information directly from the documents.

---

## ✨ Features

### Core Features

* Upload one or multiple PDF research papers
* Conversational question answering
* Research paper summarization
* Explain concepts in simple language
* Follow-up questions using chat history
* Page-level source citations
* Download chat transcripts

### Advanced Features

* Multi-PDF support
* Persistent FAISS vector indexes
* Hybrid Retrieval (BM25 + Semantic Search)
* MultiQueryRetriever for improved retrieval quality
* Maximum Marginal Relevance (MMR)
* Streaming responses
* Query contextualization
* Suggested questions
* Paper comparison mode
* Modular and production-ready architecture

---

## 🏗️ System Architecture

```text
PDF Upload
     ↓
PyPDFLoader
     ↓
Document Objects
     ↓
RecursiveCharacterTextSplitter
     ↓
Document Chunks
     ↓
HuggingFace Embeddings
     ↓
FAISS Vector Store
     ↓
Hybrid Retrieval
(BM25 + MMR)
     ↓
MultiQueryRetriever
     ↓
Retrieved Context
     ↓
ChatPromptTemplate
     ↓
Groq LLM
     ↓
Generated Answer
     ↓
Streamlit Interface
```

---

## 🧠 Tech Stack

### Programming Language

* Python

### Frontend

* Streamlit

### LLM Provider

* Groq API
* Llama-3.1-8B-Instant
* Llama-3.3-70B-Versatile

### Frameworks

* LangChain
* LangChain Expression Language (LCEL)

### Document Processing

* PyPDFLoader
* RecursiveCharacterTextSplitter

### Embeddings

* sentence-transformers/all-MiniLM-L6-v2
* HuggingFaceEmbeddings

### Vector Database

* FAISS

### Retrieval Techniques

* BM25 Retriever
* Maximum Marginal Relevance (MMR)
* EnsembleRetriever
* MultiQueryRetriever

### Utilities

* Python Dotenv
* Rank BM25

---

## 📁 Project Structure

```text
research_paper_chatbot/
│
├── app.py
├── config.py
├── requirements.txt
├── .env
│
├── src/
│   ├── __init__.py
│   ├── document_loader.py
│   ├── text_splitter.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── prompts.py
│   ├── rag_chain.py
│   └── utils.py
│
├── data/uploads/
├── vectorstore/
└── embedding_cache/
```

---

## 📂 Module Description

### app.py

Main Streamlit application responsible for:

* User interface
* PDF upload
* Chat interface
* Session management
* Streaming responses

### config.py

Centralized configuration file containing:

* API configuration
* Model settings
* Chunking parameters
* Retrieval parameters
* Directory paths

### document_loader.py

Responsible for:

* Loading PDFs using PyPDFLoader
* Metadata extraction
* Error handling
* Multi-PDF support

### text_splitter.py

Responsible for:

* Chunk generation
* Overlapping context preservation
* Metadata assignment

### vector_store.py

Responsible for:

* Embedding generation
* FAISS index creation
* Vector store persistence
* Index loading

### retriever.py

Responsible for:

* Hybrid retrieval
* BM25 retrieval
* MMR retrieval
* MultiQuery retrieval pipeline

### prompts.py

Responsible for:

* System prompts
* Contextualization prompts
* Paper comparison prompts

### rag_chain.py

Responsible for:

* Query rewriting
* Retrieval pipeline
* Context formatting
* Answer generation chains

### utils.py

Responsible for:

* Suggested questions
* Chat transcript generation
* Citation formatting

---

## 🔍 Retrieval Pipeline

The system uses a multi-stage retrieval strategy:

### Step 1: Query Contextualization

Converts follow-up questions into standalone questions.

Example:

User:

> Explain the methodology.

Rewritten Query:

> Explain the methodology proposed in the previously discussed research paper.

---

### Step 2: MultiQuery Retrieval

Generates multiple variations of the question:

Example:

Original:

> What are the contributions?

Generated Queries:

* What are the key contributions?
* What does the paper contribute?
* What are the main innovations?

This significantly improves retrieval quality.

---

### Step 3: Hybrid Retrieval

Combines:

#### BM25 Retrieval

Captures:

* Exact terms
* Acronyms
* Numbers
* Technical keywords

#### Semantic Retrieval

Captures:

* Meaning
* Similar concepts
* Related terminology

---

### Step 4: Maximum Marginal Relevance (MMR)

Selects chunks that are:

* Highly relevant
* Less redundant
* More diverse

This improves context quality and reduces duplicate information.

---

## 🧩 Prompt Engineering

The chatbot follows strict rules:

* Answer only from retrieved context
* Never hallucinate
* Cite source pages
* Mention when information is unavailable
* Explain concepts in simple language when requested

---

## 🔄 Conversational Memory

The system maintains:

* Previous questions
* Previous answers
* Follow-up context

Example:

Q1: Summarize the paper.

Q2: Explain its methodology.

Q3: What are its limitations?

The chatbot understands that "its" refers to the previously discussed paper.

---

## 📈 Why RAG Instead of Fine-Tuning?

### Fine-Tuning

❌ Expensive
❌ Requires retraining
❌ Difficult to update knowledge

### RAG

✅ Cost effective
✅ Easily updatable
✅ Grounded responses
✅ Source citations
✅ Reduced hallucinations

---

## 💻 Installation

### Clone Repository

```bash
git clone https://github.com/your-username/research-paper-chatbot.git
cd research-paper-chatbot
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Mac/Linux:

```bash
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

---

## ▶️ Run Application

```bash
streamlit run app.py
```

---

## 🎯 Example Questions

* Summarize this paper.
* Who are the authors?
* When was this paper published?
* What problem does this paper solve?
* What methodology was proposed?
* What makes this paper different from previous work?
* Explain this paper in simple language.
* Compare these papers.

---

## 🚀 Future Improvements

* Cross-Encoder Re-ranking
* Research paper recommendation system
* Knowledge graph generation
* Figure and table extraction
* Voice-based interaction
* Citation generation
* Docker deployment
* Cloud deployment with authentication

---

## 🎓 Learning Outcomes

This project demonstrates practical knowledge of:

* Retrieval-Augmented Generation (RAG)
* LangChain and LCEL
* Prompt Engineering
* Embeddings and Vector Databases
* Hybrid Retrieval Systems
* Conversational AI
* Streamlit Deployment
* Production-Style Generative AI Architecture

---

## 📌 Resume Description

Developed an industry-level Research Paper Chatbot using Retrieval-Augmented Generation (RAG), LangChain, FAISS, Hybrid Retrieval, and Groq LLMs to enable conversational question answering over research papers with source-grounded responses, conversational memory, and multi-document comparison capabilities.
