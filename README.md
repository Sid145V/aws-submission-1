<div align="center">

# AWS Customer Agreement RAG Assistant

**A production-ready Retrieval-Augmented Generation (RAG) system for legal document Q&A**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://aws-submission-1-uxpa7cch2fm32cwexhosrv.streamlit.app)
[![Backend API](https://img.shields.io/badge/Backend%20API-Render-46E3B7?style=for-the-badge&logo=render)](https://aws-rag-backend.onrender.com)
[![API Docs](https://img.shields.io/badge/Swagger-Docs-85EA2D?style=for-the-badge&logo=swagger)](https://aws-rag-backend.onrender.com/docs)

</div>

---

## 🚀 Live Demo

| Service | URL |
|---------|-----|
| **Frontend (Streamlit)** | [aws-submission-1-uxpa7cch2fm32cwexhosrv.streamlit.app](https://aws-submission-1-uxpa7cch2fm32cwexhosrv.streamlit.app) |
| **Backend API** | [aws-rag-backend.onrender.com](https://aws-rag-backend.onrender.com) |
| **Swagger API Docs** | [/docs](https://aws-rag-backend.onrender.com/docs) |

> ⚠️ Backend runs on Render's free tier — first request may take ~50 seconds to wake up.

---

## 📌 Overview

A complete, industry-grade RAG pipeline that ingests, chunks, indexes, and answers complex legal queries from the **19-page AWS Customer Agreement PDF** using Google Gemini 2.5 Flash as the LLM.

Features a **FastAPI backend**, **FAISS vector search**, **Google Generative AI embeddings**, a **custom dark-themed Streamlit chat UI**, and an **interactive Plotly analytics dashboard** — all deployed on free cloud infrastructure.

---

## ✨ Key Features

- 📄 **PDF Ingestion Pipeline** — PyMuPDF parser with recursive header-aware chunking
- 🔍 **Semantic Search** — FAISS vector store with cosine similarity scoring
- 🧠 **Gemini 2.5 Flash LLM** — Grounded answers with source citations
- 🛡️ **Hallucination Prevention** — Queries below 0.55 similarity score bypass the LLM entirely
- 💬 **Chat Assistant UI** — Custom dark-themed Streamlit interface
- 📊 **Analytics Dashboard** — Real-time Plotly charts tracking query performance
- 🗄️ **Query Logging** — SQLite database logging every query with metadata
- 🌐 **REST API** — Full FastAPI backend with Swagger documentation

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | Streamlit + Plotly | Chat UI & analytics dashboard |
| **Backend** | FastAPI + Uvicorn | High-performance REST API |
| **LLM** | Google Gemini 2.5 Flash | Answer generation |
| **Embeddings** | Google Text Embedding 004 | Semantic vector generation |
| **Vector DB** | FAISS (CPU) | Local similarity search |
| **PDF Parser** | PyMuPDF | Fast, structure-preserving extraction |
| **Database** | SQLite + SQLAlchemy | Query logging & analytics |
| **Hosting** | Render + Streamlit Cloud | Free cloud deployment |

---

## 🏗️ System Architecture

```
User Question
     │
     ▼
Streamlit Frontend
     │ POST /ask
     ▼
FastAPI Backend
     │
     ├──► FAISS Vector Store (similarity search)
     │         │
     │    Score < 0.55 ──► "Out of context" response (no LLM call)
     │    Score ≥ 0.55 ──► Gemini 2.5 Flash
     │                          │
     │                          ▼
     │                   Answer + Citations
     │
     └──► SQLite (log every query)
               │
               ▼
         Analytics Dashboard
```

---

## 🧠 Design Decisions

| Component | Choice | Why |
|-----------|--------|-----|
| **PDF Parser** | PyMuPDF | Fastest parser, preserves page structure and indentation |
| **Chunking** | Recursive (1000 chars, 200 overlap) | Prevents mid-sentence cuts, preserves context |
| **Embeddings** | Google Text Embedding 004 | Free API, no RAM overhead vs local models |
| **Vector DB** | FAISS | Lightweight, local, zero cloud dependency |
| **RAG Threshold** | 0.55 cosine similarity | Stops hallucinations for out-of-scope queries |
| **LLM** | Gemini 2.5 Flash | Fast, large context window, free tier available |
| **Database** | SQLite | Zero config, embedded, perfect for analytics |
| **Frontend** | Streamlit | Rapid Python UI with Plotly chart integration |

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check & system status |
| POST | `/ingest` | Ingest & index the AWS PDF |
| POST | `/ask` | Ask a legal question, get RAG answer |
| GET | `/analytics` | Query performance aggregations |

### Example — Ask a question:
```bash
curl -X POST "https://aws-rag-backend.onrender.com/ask" \
     -H "Content-Type: application/json" \
     -d '{"query": "How can I terminate the agreement?"}'
```

---

## 📁 Project Structure

```
project/
├── backend/
│   ├── api/
│   │   ├── ask.py            # RAG Q&A endpoint
│   │   ├── ingest.py         # PDF ingestion endpoint
│   │   └── analytics.py      # Analytics aggregation endpoint
│   ├── services/
│   │   ├── embeddings.py     # Google embedding wrapper
│   │   ├── vector_store.py   # FAISS manager & similarity search
│   │   ├── rag_pipeline.py   # RAG pipeline coordinator
│   │   ├── chunker.py        # Recursive text chunker
│   │   └── pdf_loader.py     # PyMuPDF PDF parser
│   ├── database/
│   │   ├── db.py             # SQLite connection
│   │   └── models.py         # QueryLog SQLAlchemy model
│   └── main.py               # FastAPI app entry point
├── frontend/
│   └── app.py                # Streamlit Chat UI + Analytics
├── data/
│   └── aws_customer_agreement.pdf
└── requirements.txt
```

---

## 🚀 Local Setup

### Prerequisites
- Python 3.10+
- Google Gemini API Key — free from [aistudio.google.com](https://aistudio.google.com)

### Run locally

```bash
# Clone the repo
git clone https://github.com/Sid145V/aws-submission-1.git
cd aws-submission-1/project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Add your GEMINI_API_KEY to .env

# Start backend
uvicorn backend.main:app --host 127.0.0.1 --port 8000

# Start frontend (new terminal)
streamlit run frontend/app.py
```

Then open [http://localhost:8501](http://localhost:8501)

---

## 🌟 What Makes This Stand Out

- **End-to-end RAG pipeline** built from scratch — not a tutorial copy
- **Hallucination guard** — similarity threshold prevents false answers
- **Analytics dashboard** — tracks system performance in real time
- **Production deployed** on free cloud infrastructure
- **Clean API design** with full Swagger documentation

---

## 👨‍💻 Author

**Siddanagouda** — B.E. Computer Science, CBIT Kolar (2026)

Targeting roles in **AI Full Stack**, **GenAI Engineering**, and **Full Stack Development**.

[![GitHub](https://img.shields.io/badge/GitHub-Sid145V-181717?style=flat&logo=github)](https://github.com/Sid145V)

---

<div align="center">
  <sub>Built as part of VeStaff Junior AI Developer Technical Assignment</sub>
</div>
