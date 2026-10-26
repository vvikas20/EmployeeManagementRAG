# 🏢 Enterprise Employee Management RAG & Multi-Agent System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![SQLite](https://img.shields.io/badge/SQLite-Zero--Dependency-003B57.svg)](https://sqlite.org)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-FF6F00.svg)](https://www.trychroma.com)
[![LiteLLM](https://img.shields.io/badge/LiteLLM-Groq_%7C_Gemini_%7C_Ollama-7B1FA2.svg)](https://litellm.ai)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com)
[![Tests Passing](https://img.shields.io/badge/Tests-27%2F27_Passed-brightgreen.svg)]()

> A production-grade **Enterprise Multi-Agent Assistant** combining **structured relational database queries (SQLite)**, **unstructured semantic policy retrieval (ChromaDB RAG)**, and **automated business actions (Email Drafting)** with real-time SSE streaming and execution trace visualization.

---

## 🌟 Key Highlights

- **Zero-Dependency Database**: Powered by Python's built-in **SQLite** (`./data/employee_management.db`). Auto-initializes and auto-seeds with **28 realistic enterprise employees** across **8 departments**, **24 designations**, **5 locations**, and **7 leave types**.
- **Comprehensive Corporate Policy RAG**: Embedded **ChromaDB** vector store automatically chunked and indexed with the **10-section Corporate Policy Manual** (`sample_policy.md`), covering core hours (10 AM–4 PM), leave accruals, health coverage, travel per diems, and certificate pickups at the HQ reception desk.
- **High-Speed Cloud & Offline LLM Support**: Unified **LiteLLM** client configured for:
  - **Groq API** (Active): Ultra-fast inference with `openai/gpt-oss-120b` (synthesis & SQL) and `openai/gpt-oss-20b` (routing & intent).
  - **Google Gemini**: Hosted cloud inference (`gemini-2.5-flash`).
  - **Local Ollama**: 100% offline local inference (`llama3.1`, `qwen2.5:1.5b`) with `mxbai-embed-large` embeddings.
- **Multi-Agent Orchestration & Memory**:
  - **Memory Agent**: Multi-turn context tracking, token budgeting (~1000 tokens), and pronoun resolution (e.g. mapping "write him an email" to `EMP4001`) without redundant clarification loops.
  - **Supervisor Router**: Classifies queries into 6 execution tracks via structured JSON.
  - **Tool Calling Agent**: Parameterized SQL tools (`get_employee_profile`, `get_leave_balance`, `get_team_members`, `get_department_stats`).
  - **Text-to-SQL Agent**: Safely generates arbitrary SQLite queries guarded by `sqlglot` AST validation (enforcing `SELECT`-only and `LIMIT 50`).
  - **Email Agent**: Synthesizes formal corporate communications with automatic manager CC lookups.
- **Docker Containerization**: Instant container deployment via Docker Compose with persistent data volumes.
- **Model Context Protocol (MCP)**: Native stdio server (`mcp_server.py`) for VS Code GitHub Copilot and Claude Desktop integration.
- **Modern Responsive Web UI**: Real-time Server-Sent Events (SSE) progressive Markdown rendering with a collapsible live Agent Execution Trace Visualizer.

---

## 🚀 Quick Start

### Option A: Run Pre-Built Image from Docker Hub (Fastest)

```powershell
docker run -d -p 8000:8000 `
  --name employee-rag-app `
  -e GROQ_API_KEY="your-groq-api-key" `
  vvikas20/employee-rag-system:latest
```

### Option B: Run with Docker Compose

```powershell
# 1. Provide your Groq or Gemini API key in config.local.yaml
# groq:
#   api_key: "gsk_..."

# 2. Build and launch
docker compose up --build -d

# 3. View logs
docker compose logs -f
```

Open your browser to:
- 🌐 **Web UI:** [http://localhost:8000](http://localhost:8000)
- 📡 **Health Check:** [http://localhost:8000/api/health](http://localhost:8000/api/health)
- 📖 **Swagger API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

### Option B: Run Locally (Native Python)

```powershell
# 1. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Linux/macOS: source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch application server
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

---

## 🧪 Running Automated Tests

Run the full 27-test verification suite:

```powershell
.venv\Scripts\python.exe -m pytest tests/ -v
```

All 27 tests pass across SQL tools, memory agent, SSE streaming, and SQLite database safety checks.

---

## 📚 Documentation Directory

Explore the detailed architecture guides, learning resources, and operational manuals:

| Document | Description |
| :--- | :--- |
| 📖 [**SETUP_AND_RUN.md**](./SETUP_AND_RUN.md) | Comprehensive installation, Docker configuration, and LLM profile setup guide. |
| 🏗️ [**SOLUTION_ARCHITECTURE.md**](./SOLUTION_ARCHITECTURE.md) | In-depth system architecture, sequence diagrams, and security guardrail details. |
| 📊 [**architecture_report.md**](./architecture_report.md) | Component breakdowns, agent contracts, and infrastructure topologies. |
| 🎓 [**AI_ML_LEARNING_GUIDE.md**](./AI_ML_LEARNING_GUIDE.md) | Educational primer explaining GenAI, RAG, embeddings, agents, and tokens with direct links to repo code. |
| 📜 [**sample_policy.md**](./sample_policy.md) | 10-section enterprise corporate policy manual indexed in ChromaDB. |

---

## 🔒 Security & Privacy Features

- **Strict Read-Only Enforcement**: Text-to-SQL statements are validated with `sqlglot` AST parsing. Mutation statements (`INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, etc.) are rejected before execution.
- **Sensitive Data Redaction**: The schema exposed to the LLM automatically excludes sensitive columns like `Salary`.
- **Injection-Proof Tool Calling**: Core profile lookups use parameterized SQL queries (`?` parameter binding).
- **Credential Isolation**: API keys reside exclusively in git-ignored `config.local.yaml`.
