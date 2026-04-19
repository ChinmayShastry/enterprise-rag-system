# 🤖 Enterprise-Grade RAG System — Company Knowledge AI Assistant

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?logo=openai)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?logo=streamlit)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A production-ready Retrieval-Augmented Generation (RAG) system that allows users to query company documents using natural language. Built with hybrid search, role-based access control, query rewriting, query logging, user feedback, and an evaluation dashboard.

> **Not a "chat with PDF" toy.** This is an enterprise-grade GenAI backend with real retrieval engineering, access control, and observability.

---

## 📸 Demo

| Login Screen | Chat Interface | Admin Dashboard |
|---|---|---|
| Role-based login | Answers with citations | Query logs + feedback metrics |

---

## 🧠 Architecture

```
PDF / Scanned Document
        ↓
   OCR (Tesseract)
        ↓
   Text Chunking (RecursiveCharacterTextSplitter)
        ↓
   OpenAI Embeddings (text-embedding-3-small)
        ↓
   ChromaDB Vector Store
        ↓
User Query → RBAC Check → Query Rewriter (GPT-4o-mini)
                                    ↓
              ┌─────────────────────────────────────┐
              │  Vector Search    +   BM25 Search   │
              │  (Semantic)           (Keyword)     │
              └─────────────────────────────────────┘
                                    ↓
                         Deduplicated Results
                                    ↓
                    GPT-4o-mini → Answer + Citations
                                    ↓
                    👍 / 👎 Feedback → Query Logger
                                    ↓
                         Evaluation Dashboard
```

---

## ✨ Features

### 🔍 Retrieval
- **Hybrid Search** — combines ChromaDB vector search (semantic) with BM25 (keyword) for best-of-both retrieval
- **Query Rewriting** — GPT rewrites each user query into 4 variations before searching, drastically reducing missed results
- **Deduplication** — merges results from all query variations without repetition

### 🔐 Access Control (RBAC)
- Three user roles: `admin`, `support`, `viewer`
- Each role has different retrieval depth and source visibility
- Unauthenticated users cannot access the system

| Role | Max Chunks | See Sources |
|---|---|---|
| Admin | 8 | ✅ Yes |
| Support | 5 | ✅ Yes |
| Viewer | 3 | ❌ Hidden |

### 📋 Observability
- Every query is logged with timestamp, username, role, question, answer, and sources
- User feedback (👍 / 👎) saved to a separate log
- Admin-only evaluation dashboard shows query stats, satisfaction scores, and system health

### 🖥️ Frontend
- Clean Streamlit UI with login screen, chat interface, and admin dashboard
- Chat history preserved within session
- Feedback buttons on every answer

---

## 🏗️ Tech Stack

| Component | Technology |
|---|---|
| LLM | OpenAI GPT-4o-mini |
| Embeddings | OpenAI text-embedding-3-small |
| Vector Store | ChromaDB |
| Keyword Search | BM25 (rank-bm25) |
| OCR | Tesseract + pdf2image |
| Document Loading | LangChain |
| Frontend | Streamlit |
| Language | Python 3.10+ |

---

## 📁 Project Structure

```
RAG_Project/
├── app.py                   # Main Streamlit application
├── chroma_db/               # Persisted vector store (ChromaDB)
├── logs/
│   ├── query_log.jsonl      # All user queries + answers
│   └── feedback_log.jsonl   # User feedback (useful / not useful)
├── data/
│   └── your_document.pdf    # Source document
├── RAG_Phase1.ipynb         # Colab notebook (full pipeline dev)
├── requirements.txt         # Python dependencies
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/enterprise-rag-system.git
cd enterprise-rag-system
```

### 2. Create and activate virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your OpenAI API key

Open `app.py` and replace:

```python
OPENAI_API_KEY = "your-openai-api-key-here"
```

> 💡 Tip: Use environment variables in production — never hardcode API keys.

### 5. Add your document and build the vector store

If you want to use your own PDF, run the Colab notebook `RAG_Phase1.ipynb` end-to-end. It will:
- OCR your PDF (if scanned)
- Chunk the text
- Generate embeddings
- Save the ChromaDB vector store to Google Drive

Then download `chroma_db/` from Drive and place it in the project root.

### 6. Run the app

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 👤 Demo Accounts

| Username | Password | Role |
|---|---|---|
| alice | alice123 | admin |
| bob | bob123 | support |
| guest | guest123 | viewer |

---

## 📦 Requirements

```
openai
chromadb
pypdf
langchain
langchain-community
langchain-openai
tiktoken
sentence-transformers
rank_bm25
streamlit
pytesseract
pdf2image
pymupdf
```

Install all at once:

```bash
pip install openai chromadb pypdf langchain langchain-community langchain-openai tiktoken sentence-transformers rank_bm25 streamlit pytesseract pdf2image pymupdf
```

---

## 🔬 How It Works — Deep Dive

### Step 1: Document Ingestion
The system uses `pdf2image` + Tesseract OCR to extract text from scanned PDFs. Text-based PDFs are handled by `PyPDFLoader`. Each page is converted to a `Document` object with metadata (page number, source).

### Step 2: Chunking
Documents are split using `RecursiveCharacterTextSplitter` with:
- `chunk_size = 500` characters
- `chunk_overlap = 100` characters (ensures context continuity across chunks)
- Split priority: paragraph → line → sentence → word

### Step 3: Embeddings + Vector Store
All chunks are embedded using OpenAI's `text-embedding-3-small` model and stored in ChromaDB. The vector store is persisted to disk so re-embedding is not required on restart.

### Step 4: Query Pipeline

When a user submits a question:

1. **RBAC check** — verify user role and permissions
2. **Query rewriting** — GPT generates 3 alternative phrasings of the question
3. **Hybrid search** — run all 4 queries (original + 3 rewrites) through:
   - ChromaDB vector search (semantic similarity)
   - BM25 keyword search (exact term matching)
4. **Deduplication** — merge all results, remove duplicates
5. **Answer generation** — top chunks are passed to GPT-4o-mini as context
6. **Logging** — query, answer, user info, and sources saved to JSONL log

### Why Hybrid Search?
Vector search is great for semantic similarity ("machine shaking") but misses exact keyword matches. BM25 is great for exact terms but misses synonyms. Combining both gives the best retrieval coverage — this is what production RAG systems like Elasticsearch use.

### Why Query Rewriting?
Users often phrase questions differently from how documents are written. A user asks "vibrating too much" but the manual says "not installed on a level floor." Query rewriting generates alternative phrasings using GPT before searching, bridging this vocabulary gap.

---

## 📊 Evaluation Dashboard (Admin Only)

The admin dashboard shows:
- Total queries logged
- Total feedback received
- 👍 Useful vs 👎 Not useful counts
- Overall satisfaction score (%)
- Last 10 queries with full details (expandable)
- System health (vector DB size, models used, log file paths)

---

## 🗺️ Development Phases

| Phase | What was built |
|---|---|
| Phase 1 | OCR pipeline, chunking, embeddings, ChromaDB, basic RAG with citations |
| Phase 2 | Query rewriting, hybrid search (Vector + BM25), hallucination reduction |
| Phase 3 | RBAC login system, role-controlled retrieval, query logging |
| Phase 4 | User feedback system, evaluation dashboard, Streamlit frontend |

---

## 🧪 Example Queries

```
"How do I clean the filter?"
→ Step-by-step cleaning instructions (Page 31)

"What should I do if the machine is vibrating too much?"
→ Check fixing bolts and floor leveling (Page 34)

"What washing programs are available?"
→ Full list: Cotton, Synthetic, Wool, ECO 40-60, Steam Treat... (Page 25)
```

---

## 🔮 Future Improvements

- [ ] FastAPI backend (decouple frontend and retrieval logic)
- [ ] Multi-document support with document-level access control
- [ ] Elasticsearch integration for production-grade BM25
- [ ] Graph RAG for multi-hop reasoning across documents
- [ ] LLM evaluation metrics (faithfulness, answer relevance, context precision)
- [ ] Docker containerization for deployment
- [ ] JWT-based authentication (replace hardcoded users)
- [ ] Upload new documents via UI without re-running notebook

---

## 🎯 Why This Project Matters

Most RAG tutorials build a basic "chat with PDF" using a 10-line LangChain chain. This project goes further:

| What tutorials show | What this project builds |
|---|---|
| Basic vector search | Hybrid search (vector + BM25) |
| One query per search | Query rewriting (4 variations) |
| No access control | RBAC with 3 roles |
| No logging | Full query + feedback logging |
| No observability | Admin evaluation dashboard |
| Notebook only | Deployed Streamlit app |

This reflects how RAG systems are actually built and deployed in production.

---

## 👨‍💻 Author

**Chinmay**
- 📍 Bengaluru, India

---

## 📄 License

This project is licensed under the MIT License.
