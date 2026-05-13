# 📖 DocuMind — AI Assistant for Product Manuals & Compliance Documents

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?logo=openai)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?logo=streamlit)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-orange)
![License](https://img.shields.io/badge/License-MIT-green)

> **The problem:** Product manuals are dense, scanned, version-specific, and safety-critical. Compliance documents are long, clause-heavy, and legally binding. Support teams waste hours searching through them manually. A wrong answer from a generic AI about a torque spec or a compliance clause is not just unhelpful — it is dangerous.

**DocuMind** is a RAG system built specifically for two document types where getting the answer wrong has real consequences — **product manuals** and **HR/compliance documents**. It retrieves answers only from your actual documents, cites exact pages, respects who is allowed to see what, and logs every query for audit purposes.

This is not a "chat with any PDF" tool. It is an enterprise assistant built around the specific retrieval, safety, and access-control needs of technical and compliance documentation.

---

## 🎯 Built for Two Specific Problems

### 📦 Product Manuals

Field technicians, support agents, and end users need fast, accurate answers from manuals that are often scanned, image-heavy, and hundreds of pages long.

| What users ask | What DocuMind does |
|---|---|
| "How do I fix error code E4?" | Retrieves the exact troubleshooting section, cites the page |
| "What is the max load for this model?" | Returns the actual specification — not a guess |
| "Walk me through installation step by step" | Returns ordered steps with safety warnings surfaced first |
| Support agent vs end customer need different depth | RBAC controls retrieval depth and source visibility per role |

### 📋 HR & Compliance Documents

HR managers, legal teams, and employees need exact policy answers with traceable citations — not paraphrased summaries that could misrepresent a clause.

| What users ask | What DocuMind does |
|---|---|
| "What is the notice period for contract staff?" | Retrieves the exact clause, not a paraphrase |
| "Which section covers data retention?" | Cites section number and page |
| "Who asked what, and when?" | Every query logged — audit trail built in |
| Sensitive HR data cannot be seen by all staff | Role-based access hides sources from lower-privilege users |

---

## Why Not Just Use ChatGPT?

| ChatGPT / Generic LLM | DocuMind |
|---|---|
| Hallucinates part numbers, torque specs, policy clauses | Answers only from your actual document — if it is not there, it says so |
| No page citations | Every answer cites exact page numbers |
| No knowledge of your specific product version or internal policy | Ingested per product, per document version |
| Cannot read scanned product manuals (image PDFs) | OCR pipeline handles image-only PDFs automatically |
| Anyone can ask anything and see everything | RBAC — field tech, support, and admin see different levels of detail |
| No audit trail | Full query log — who asked what, when, and what answer they received |
| Trained on public internet, may be outdated for your product | Grounded entirely in your current documentation |

---

## Technical Features

| Feature | Details |
|---|---|
| **OCR Pipeline** | Auto-detects scanned PDFs and runs Tesseract — handles real-world product manuals |
| **Hybrid Search** | ChromaDB vector search (semantic) + BM25 keyword search — best of both retrieval methods |
| **Query Rewriting** | GPT rewrites each query into 3 variations, bridging the vocabulary gap between users and documents |
| **Cross-Encoder Reranking** | ms-marco-MiniLM-L-6-v2 scores chunks by true relevance, not just cosine similarity |
| **Streaming Answers** | Token-by-token streaming — no blank wait screen |
| **RBAC** | 3 roles (admin / support / viewer) with configurable retrieval depth per role |
| **Query Logging** | Every query, answer, user, and timestamp saved — compliance-ready audit trail |
| **Feedback System** | Per-answer thumbs up / down, logged separately for evaluation |
| **Evaluation Dashboard** | Satisfaction scores, usage by role, recent query review — admin only |
| **Fully Configurable** | Title, persona, chunk settings, models — all in one YAML file, no code changes needed |

---

## Architecture

```
PDF / Scanned Manual
        |
   [Auto-detect]
   /           \
PyPDF        OCR (Tesseract)
        |
  Chunking (500 chars, 100 overlap)
        |
  OpenAI Embeddings
        |
  ChromaDB  +  BM25 Index
        |
User Query --> RBAC Check --> Role sets retrieval depth
        |
  Query Rewriter (3 alternative phrasings)
        |
  Hybrid Search (Vector + BM25, all 4 queries)
        |
  Deduplicate
        |
  Cross-Encoder Reranker
        |
  Top-N Chunks (per role limit)
        |
  GPT-4o-mini (streaming)
        |
  Answer + Page Citations
  /               \
Query Log       Feedback Log
        \           /
       Admin Dashboard
```

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/ChinmayShastry/enterprise-rag-system.git
cd enterprise-rag-system
pip install -r requirements.txt

# 2. Set your API key
cp .env.example .env
# Open .env and add: OPENAI_API_KEY=sk-your-key-here

# 3. Ingest your document
python scripts/ingest.py --pdf data/your_manual.pdf

# 4. Run
streamlit run app.py
```

Open `http://localhost:8501`. Log in with any demo account and start asking questions.

---

## Adapting to Your Document

Everything lives in `config/config.yaml` — no code changes needed.

**For a product manual:**
```yaml
app:
  title: "Bosch Dishwasher Support Assistant"
  icon: "🍽️"
  persona: "a support assistant for Bosch dishwasher product manuals"
  description: "Ask anything about installation, troubleshooting, and maintenance."

rag:
  collection_name: "bosch_dishwasher_manual"
  chunk_size: 500
  chunk_overlap: 100
```

**For an HR policy document:**
```yaml
app:
  title: "Acme HR Policy Assistant"
  icon: "📋"
  persona: "an HR assistant for Acme company policies and compliance documents"
  description: "Ask about leave policies, compliance procedures, and employee guidelines."

rag:
  collection_name: "acme_hr_policies"
  chunk_size: 600
  chunk_overlap: 150
```

Switch documents by updating the config and re-ingesting:

```bash
python scripts/ingest.py --pdf data/new_document.pdf --reset
```

---

## Ingestion Options

```bash
# Standard — auto-detects text vs scanned
python scripts/ingest.py --pdf data/manual.pdf

# Force OCR for scanned or image-heavy PDFs
python scripts/ingest.py --pdf data/scanned_manual.pdf --ocr

# Replace existing collection with a new version
python scripts/ingest.py --pdf data/manual_v2.pdf --reset
```

The script automatically falls back to OCR if text extraction returns mostly empty pages — common with older product manuals.

---

## Roles and Access Control

Defined in `config/users.yaml` and `config/config.yaml`.

| Role | Chat | Dashboard | Retrieval depth | See page citations |
|---|---|---|---|---|
| `admin` | Yes | Yes | 5 chunks from 8 candidates | Yes |
| `support` | Yes | No | 3 chunks from 5 candidates | Yes |
| `viewer` | Yes | No | 2 chunks from 3 candidates | No |

**Why does retrieval depth matter?**
An admin or support agent debugging a complaint needs full context. An end customer asking a basic question needs a clean, focused answer — not 8 overlapping chunks of manual text. Role-based depth controls both answer quality and information exposure.

> For production: replace plain-text passwords in `users.yaml` with a proper identity provider (Auth0, Cognito, LDAP).

---

## How the Retrieval Pipeline Works

### The vocabulary gap problem

A user asks: *"machine is shaking badly"*
The manual says: *"excessive vibration during spin cycle — check levelling feet"*

Same meaning, zero shared keywords. Pure vector search catches this. Pure keyword search (BM25) misses it entirely.

But if a user asks: *"error code E4"* — an exact code — vector search may drift to semantically similar but wrong results. BM25 catches exact codes reliably.

Hybrid search runs both and merges results, giving strong recall for both semantic and keyword queries.

### Query rewriting

Before searching, GPT generates 3 alternative phrasings. All 4 variations run through hybrid search. This closes the gap between casual user language and formal technical document language.

### Cross-encoder reranking

After retrieval, 20+ chunks may be returned. A cross-encoder model reads the question and each chunk together — the way a human would — and assigns a proper relevance score. Only the top-N per role go to the LLM. This is the step that most "chat with PDF" tools skip, and it is what directly reduces hallucination by cutting noisy context.

---

## Project Structure

```
enterprise-rag-system/
├── app.py                    # Streamlit frontend
├── rag/
│   ├── __init__.py
│   ├── auth.py               # Authentication and RBAC
│   ├── retrieval.py          # Hybrid search and reranking
│   ├── generation.py         # Streaming answer generation
│   └── logger.py             # Query and feedback logging
├── scripts/
│   └── ingest.py             # PDF ingestion pipeline (run once per document)
├── config/
│   ├── config.yaml           # App config, RAG settings, role permissions
│   └── users.yaml            # User credentials and roles
├── notebooks/
│   └── RAG_development.ipynb # Development notebook (Phases 1-4)
├── logs/                     # gitignored — created at runtime
├── data/                     # gitignored — put your PDF here
├── chroma_db/                # Vector store — committed for demo deployment
├── .env.example              # Copy to .env, add your API key
├── .gitignore
├── requirements.txt
└── packages.txt              # System deps for Streamlit Cloud
```

---

## Development Phases

| Phase | What was built |
|---|---|
| Phase 1 | OCR pipeline, chunking, OpenAI embeddings, ChromaDB, basic RAG with page citations |
| Phase 2 | Query rewriting (3 variations), hybrid search (vector + BM25), retrieval recall improvements |
| Phase 3 | RBAC login system, role-controlled retrieval depth, query logging to JSONL |
| Phase 4 | Cross-encoder reranking, streaming answers, feedback system, evaluation dashboard |
| Phase 5 | Modular refactor (rag/ package), config-driven design, standalone ingest.py script |

---

## Roadmap

- [ ] Multi-document support — query across your entire product catalog, filter by product
- [ ] Safety warning pinning — chunks with WARNING or CAUTION always surface first
- [ ] Document versioning — tag manuals by product model and version, prevent cross-version bleed
- [ ] RAGAS evaluation — automated faithfulness, answer relevance, and context precision scoring
- [ ] FastAPI backend — decouple retrieval from Streamlit for API access by other systems
- [ ] Docker deployment — single command local or cloud setup
- [ ] JWT authentication — replace the YAML credential system

---

## Author

**Chinmay Shastry** — Bengaluru, India
[GitHub](https://github.com/ChinmayShastry)

---

## License

MIT
