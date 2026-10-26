# 🏢 Enterprise Employee Management RAG & Multi-Agent System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![SQLite](https://img.shields.io/badge/SQLite-Zero--Dependency-003B57.svg)](https://sqlite.org)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-FF6F00.svg)](https://www.trychroma.com)
[![LiteLLM](https://img.shields.io/badge/LiteLLM-Groq_%7C_Gemini_%7C_Ollama-7B1FA2.svg)](https://litellm.ai)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com)
[![Tests Passing](https://img.shields.io/badge/Tests-27%2F27_Passed-brightgreen.svg)]()

> A production-grade **Enterprise Multi-Agent Assistant** combining **structured relational database queries (SQLite)**, **unstructured semantic policy retrieval (ChromaDB RAG)**, and **automated corporate actions (Email Generation)** with real-time SSE streaming, interactive step debugging, and a minimal enterprise execution trace UI.

---

## 🌟 Key Highlights

- **Zero-Dependency Relational Database**: Powered by Python's built-in **SQLite** (`./data/employee_management.db`). Automatically initializes and seeds with **28 realistic enterprise employees** across **8 departments**, **24 designations**, **5 locations**, and **7 leave types**.
- **Comprehensive Corporate Policy RAG**: Persistent **ChromaDB** vector store automatically chunked and indexed with the **10-section Corporate Policy Manual** (`sample_policy.md`), indexing core hours (10 AM–4 PM), leave entitlements, health benefits, travel per diems, and HR protocols.
- **High-Speed Cloud & Offline LLM Support**: Unified **LiteLLM** client configured for seamless switching:
  - **Groq API** (Active): Ultra-fast inference with `openai/gpt-oss-120b` (complex reasoning & synthesis) and `openai/gpt-oss-20b` (intent classification & routing).
  - **Google Gemini**: Cloud inference with `gemini-3.8-flash` and `gemini-3.5-flash-lite`.
  - **Local Ollama**: 100% offline local inference (`llama3.1`, `qwen2.5:1.5b`) with `mxbai-embed-large` embeddings.
- **Multi-Agent Orchestration & Memory**:
  - **Memory Agent**: Multi-turn conversational memory, token budgeting (~1000 tokens), and pronoun/coreference resolution (e.g. mapping "write an email to her" $\rightarrow$ `Divya Menon (EMP8000)`).
  - **Supervisor Router**: Categorizes queries into 6 execution tracks via structured JSON routing.
  - **Tool Calling Agent**: Autonomous 4-iteration tool loop with parameterized SQL tools (`get_employee_profile`, `get_leave_balance`, `get_team_members`, `get_department_stats`).
  - **Text-to-SQL Agent**: Safely generates arbitrary SQLite queries guarded by `sqlglot` AST validation (enforcing `SELECT`-only and `LIMIT 50`).
  - **Email Synthesizer Agent**: Drafts executive-ready corporate emails with automatic manager CC lookups and one-click clipboard copying.
- **Minimalist Enterprise Web UI**: Clean, light SaaS design (slate/zinc palette `#f8fafc` / `#0f172a`), real-time Server-Sent Events (SSE) progressive token streaming, suggested prompt chips, and a live Multi-Agent Execution Trace visualizer.
- **Interactive Step Debugging**: Pre-configured VS Code / Antigravity debug launch profiles (`.vscode/launch.json`) for stepping through each agent breakpoint (`F5` / `F10` / `F11`).
- **Model Context Protocol (MCP)**: Native stdio server (`mcp_server.py`) for VS Code Copilot, Cursor, and Claude Desktop integration.
- **Docker Containerization**: Instant container deployment via Docker Compose with host volume persistence for SQLite and ChromaDB.

---

## 🚀 Quick Start

### Option A: Run Locally (Native Python)

```powershell
# 1. Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Linux/macOS: source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your Groq or Gemini API key in config.local.yaml
# llm:
#   profiles:
#     groq:
#       api_key: "gsk_..."

# 4. Launch application server with auto-reload
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

---

### Option B: Run with Docker Compose

```powershell
# Build and launch services in background
docker compose up --build -d

# Follow application logs
docker compose logs -f
```

Open your browser to:
- 🌐 **Web UI:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
- 📡 **Health Check:** [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)
- 📖 **Swagger API Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🔍 How to Debug Step-by-Step

You can set breakpoints and observe each agent in action:

1. In VS Code / Antigravity, open **Run & Debug** (`Ctrl + Shift + D`).
2. Select **`FastAPI (main.py)`** and press **`F5`**.
3. Set breakpoints in:
   - `agents/memory_agent.py` line 56 (`rewrite_query`)
   - `agents/router_agent.py` line 17 (`route_query`)
   - `agents/orchestrator.py` line 408 (`process_query_stream`)
   - `agents/sql_tools.py` line 165 (`dispatch`)
   - `agents/policy_agent.py` line 17 (`answer_policy_query`)
   - `agents/email_agent.py` line 12 (`generate_email`)
4. Submit any prompt in the Web UI at [http://127.0.0.1:8000](http://127.0.0.1:8000) and step through using **`F10`** (Step Over) and **`F11`** (Step Into).

---

## 🧪 Running Automated Tests

Run the complete 27-test verification suite:

```powershell
.venv\Scripts\python.exe -m pytest tests/ -v
```

**Test Coverage Highlights:**
- `tests/test_phase1.py` (13 tests): Autonomous SQL tool calling, parameter extraction, fallback error handling.
- `tests/test_phase2.py` (4 tests): Conversational memory, pronoun resolution, context summarization.
- `tests/test_phase3.py` (1 test): SSE token-by-token streaming and event schemas.
- `tests/test_sqlite_db.py` (9 tests): SQLite schema generation, safe schema projection, company statistics, leave balances, read-only AST safety validation.

---

## 📚 Documentation Directory

Explore the complete technical manuals and guides:

| Document | Description |
| :--- | :--- |
| 📖 [**SETUP_AND_RUN.md**](./SETUP_AND_RUN.md) | Comprehensive installation, Docker configuration, and local debugging guide. |
| 🏗️ [**SOLUTION_ARCHITECTURE.md**](./SOLUTION_ARCHITECTURE.md) | In-depth system architecture, multi-agent sequence diagrams, and security guardrails. |
| 📊 [**architecture_report.md**](./architecture_report.md) | Detailed component breakdowns, agent contracts, and infrastructure topologies. |
| 🎓 [**AI_ML_LEARNING_GUIDE.md**](./AI_ML_LEARNING_GUIDE.md) | Primer explaining GenAI, RAG, embeddings, agents, and tokens with direct links to repo code. |
| 📜 [**sample_policy.md**](./sample_policy.md) | 10-section enterprise corporate policy manual indexed in ChromaDB. |

---

## 🔒 Security & Guardrails

- **Strict Read-Only Enforcement**: Text-to-SQL statements are validated with `sqlglot` AST parsing. Mutation statements (`INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, etc.) are rejected before execution.
- **Sensitive Data Protection**: Parameterized tools protect internal schema integrity and prevent SQL injection.
- **Credential Isolation**: API keys reside exclusively in git-ignored `config.local.yaml`.
