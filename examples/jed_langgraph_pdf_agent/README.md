# Jed’s LangGraph Agent over PDFs (OpenAI/vLLM compatible)

A minimal **LangChain + LangGraph** agent that answers questions over your PDFs:
- Ingests PDFs → builds a local **FAISS** vector index
- Graph has a **Router → Retrieve → Answer** loop with a tool call
- Uses **OpenAI-compatible** chat endpoint (OpenAI or your vLLM server)

> Pair this with the vLLM repo you already set up to run locally.

---

## Folder
```
examples/jed_langgraph_pdf_agent/
├── README.md
├── env.yml
├── ingest_pdfs.py
├── agent_graph.py
├── run_agent.py
└── sample_questions.jsonl
```

## Quickstart

### 1) Create env
```bash
conda env create -f examples/jed_langgraph_pdf_agent/env.yml
conda activate lc-langgraph
```

### 2) Ingest PDFs → FAISS index
Put PDFs under `./pdfs/`, then:
```bash
python examples/jed_langgraph_pdf_agent/ingest_pdfs.py   --input-dir ./pdfs   --index-dir ./faiss_index
```

### 3) Set OpenAI-compatible endpoint
```bash
# OpenAI
export OPENAI_API_KEY=sk-...
export OPENAI_BASE_URL=https://api.openai.com/v1
export OPENAI_MODEL=gpt-4o-mini

# or vLLM
export OPENAI_API_KEY=dummy
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_MODEL=llama-3.1-8b-instruct
```

### 4) Run the agent (single)
```bash
python examples/jed_langgraph_pdf_agent/agent_graph.py   --index-dir ./faiss_index   --question "Summarize the main contributions discussed in section 2."
```

### 5) Batch run from JSONL
```bash
python examples/jed_langgraph_pdf_agent/run_agent.py   --index-dir ./faiss_index   --questions-file examples/jed_langgraph_pdf_agent/sample_questions.jsonl   --out runs/langgraph_batch
```

Outputs include per-question answers with cited chunks.

---

## Notes
- Embeddings: `text-embedding-3-small` by default (OpenAI-compatible). You can switch to local `sentence-transformers` by editing `ingest_pdfs.py`.
- The graph keeps things simple (no long-term memory). Extend by adding critique/retrieval loops or answer verification nodes.
