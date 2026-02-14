# Resume Chatbot (LangChain + Streamlit)

Conversational Q&A over a resume using Retrieval-Augmented Generation (RAG).
This repo provides:

- LangChain pipeline with FAISS vector store
- Conversational memory per session
- Streamlit UI with an Index Rebuild button

## 🎨 Features

- 💬 Chat interface with conversation history
- 📄 Source document viewer
- 🗑️ Clear chat history
- 📱 Responsive design
- 🎨 Custom theming

## 🎯 Quick Start

Prerequisites:
- Python 3.10+ (3.12 tested)

Setup and run locally:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Copy your resume to the repo root as resume.pdf (or set RESUME_PATH)

# Start Streamlit UI (preferred)
python -m streamlit run streamlit_app.py
```

Tip: If `streamlit` command isn't found, always use `python -m streamlit ...`.

## 🏗️ Architecture

- `app/services/rag_service.py` – Loads resume, builds vector store, exposes `query()` and `rebuild()`
- `resume_loader.py` – Loads and splits the resume PDF into chunks
- `rag_chain.py` – Builds the LangChain RAG pipeline (LLM + retriever + prompts)
- `app/services/memory_service.py` – In-memory per-session chat history
- `streamlit_app.py` – Streamlit interface (chat UI + Rebuild Index)

Data flow:
1) Resume PDF -> chunked -> embedded -> FAISS index
2) User question + chat history -> RAG chain -> answer (+ optional sources)

## 📁 File Tree (major parts)

```
.
├── streamlit_app.py             # Run Streamlit UI
├── resume_loader.py
├── rag_chain.py
├── email_sender.py
├── app/
│   ├── core/
│   │   └── config.py            # pydantic settings
│   └── services/
│       ├── rag_service.py       # RAGService
│       └── memory_service.py    # ChatMemoryService
└── .env                         # local secrets (gitignored)
```

## 💡 Configuration (.env)

This project reads settings from `.env` in the repository root. Never commit real secrets.

Example `.env` template:

```env
# Required
OPENAI_API_KEY="sk-..."
RESUME_PATH="resume.pdf"           # Path to your resume PDF

# Vector store (FAISS)
VECTORSTORE_PATH=.faiss_index       # Directory for FAISS index

# LLM / Embeddings
LLM_MODEL=gpt-4o-mini
LLM_TEMPERATURE=0.2
USE_LOCAL_EMBEDDINGS=false          # Set true only if configured

# Retrieval tuning
RETRIEVER_K=4
RETRIEVER_FETCH_K=12
MMR_LAMBDA=0.7

SESSION_TIMEOUT_MINUTES=30  # Auto-cleanup inactive sessions after 30 minutes

CLEANUP_INTERVAL_SECONDS=180  # Run cleanup every 180 seconds
```

Additional configurable settings live in `app/core/config.py` (with safe defaults): memory limits, retrieval tuning, etc.

Security note: rotate any leaked keys and keep `.env` in `.gitignore`.

## 💻 Streamlit UI

Run the app and open the chat interface:

```bash
python -m streamlit run streamlit_app.py
```

Sidebar features:
- Model and Session ID display
- Clear Chat History
- Index section showing `VECTORSTORE_PATH` and whether FAISS exists
- "Rebuild Index" button that deletes and rebuilds the FAISS index and clears chat

Automatic refresh:
- On startup, the backend computes a SHA256 of your resume. If it changed since the last run, the index is automatically rebuilt.

## 📇 Index Management (FAISS)

- Path is controlled by `VECTORSTORE_PATH` (default: `.faiss_index`).
- To force a rebuild:
	- Use the Streamlit sidebar: "Rebuild Index"
	- Or delete the directory manually and restart:
		```bash
		rm -rf .faiss_index
		```

Metadata:
- The service writes `.faiss_index/meta.json` with the resume hash and docs count for change detection.

## 🐛 Troubleshooting

- Streamlit command not found:
	```bash
	python -m streamlit run streamlit_app.py
	```

- "Resume not found" on startup:
	- Ensure `RESUME_PATH` points to an existing PDF file
	- In Codespaces/Cloud, make sure the file is present in the workspace

- Old index after updating resume:
	- Click "Rebuild Index" in Streamlit, or remove `.faiss_index/` directory

- OpenAI auth errors:
	- Set `OPENAI_API_KEY` in `.env` or environment


## Deployments

General guidance:
- Include your resume PDF in the repo (or mount it)
- Set secrets (API key, paths) in the hosting platform's secrets UI
- The first launch may take time to build the index


## Notes & Limits

- Chat memory is in-process and ephemeral. A restart clears it.
- Email notifications (`email_sender.py`) are optional; configure SMTP only if you need them.
