# 🚀 Setup & Execution Guide — Employee Management RAG System

> End-to-end instructions for running the application locally or deploying as a containerized service with Docker, supporting Groq, Google Gemini, and local Ollama LLM backends.

---

## 📋 Prerequisites Overview

| Requirement | Local Execution | Docker Container Deployment |
| :--- | :--- | :--- |
| **Operating System** | Windows 10/11, Linux, or macOS | Any OS with Docker Desktop or Docker Engine |
| **Python Runtime** | Python 3.10+ (standard library includes `sqlite3`) | Built-in via `python:3.11-slim` base image |
| **Relational Database** | **SQLite** (zero-dependency, auto-created & seeded in `./data/`) | **SQLite** (persisted via Docker host volume `./data`) |
| **Vector Database** | **ChromaDB** (embedded persistent store in `./vector_store/`) | **ChromaDB** (persisted via Docker host volume `./vector_store`) |
| **LLM Gateway** | **Groq**, **Google Gemini**, or **Local Ollama** | **Groq**, **Google Gemini**, or **Host Ollama** |

---

## 🔑 LLM Provider Profiles & API Key Configuration

The application uses **LiteLLM** to switch seamlessly between cloud and local providers. Profiles are controlled in [`config.yaml`](file:///c:/D/Playground/Gen%20AI/EmployeeManagementRAGSystem/config.yaml):

```yaml
llm:
  active_profile: "groq"   # Options: "groq", "gemini", or "ollama"
```

### Profile A: Groq (Recommended for Ultra-Fast Cloud Inference)
Active models configured:
- `default-llm`: `groq/openai/gpt-oss-120b` (Complex synthesis, SQL, and memory rewriting)
- `fast-llm`: `groq/openai/gpt-oss-20b` (High-speed intent classification & routing)

Create or update `config.local.yaml` (git-ignored) in the repository root:
```yaml
groq:
  api_key: "gsk_your_groq_api_key_here"
```

### Profile B: Google AI Studio (Gemini)
Active models configured:
- `default-llm`: `gemini/gemini-2.5-flash`
- `fast-llm`: `gemini/gemini-2.5-flash-lite`

Add your Gemini API key in `config.local.yaml`:
```yaml
gemini:
  api_key: "AIzaSy_your_gemini_api_key_here"
```

### Profile C: Local Ollama (100% Offline)
Active models:
- Chat & SQL: `llama3.1` and `qwen2.5:1.5b`
- Policy Embeddings: `mxbai-embed-large`

Ensure Ollama is running (`ollama serve`) and the required models are pulled:
```powershell
ollama pull llama3.1
ollama pull qwen2.5:1.5b
ollama pull mxbai-embed-large
```

---

## ⚡ Option 1: Running with Docker (Recommended)

Thanks to the embedded SQLite database, the system runs with zero external database dependencies.

### 1. Build and Launch
```powershell
# Start container in detached mode
docker compose up --build -d
```

### 2. Check Logs & Service Status
```powershell
# Follow application logs
docker compose logs -f
```

### 3. Access Services
- 🌐 **Web UI Dashboard:** [http://localhost:8000](http://localhost:8000)
- 📡 **API Health Status:** [http://localhost:8000/api/health](http://localhost:8000/api/health)
- 📖 **Interactive Swagger Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

### 4. Persistent Storage
Docker Compose automatically binds host mounts:
- `./data` $\rightarrow$ `/app/data` (Stores SQLite database `employee_management.db`)
- `./vector_store` $\rightarrow$ `/app/vector_store` (Stores ChromaDB vector collection)
- `./config.local.yaml` $\rightarrow$ `/app/config.local.yaml` (Supplies your Groq or Gemini API key)

---

## 💻 Option 2: Running Locally (Native Python)

### Step 1 — Create and Activate Python Virtual Environment
```powershell
# Create virtual environment
python -m venv .venv

# Activate on Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# Or on Linux / macOS:
# source .venv/bin/activate
```

### Step 2 — Install Project Dependencies
```powershell
pip install -r requirements.txt
```

*(No ODBC drivers or Microsoft SQL Server installations required; SQLite is part of standard Python).*

### Step 3 — Start the Application
```powershell
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

On initial startup:
- `./data/employee_management.db` is initialized and auto-seeded with 28 enterprise employees, 8 departments, 24 designations, and 7 leave types.
- The 10-section [`sample_policy.md`](file:///c:/D/Playground/Gen%20AI/EmployeeManagementRAGSystem/sample_policy.md) is chunked and indexed into ChromaDB.
- Navigate to [http://127.0.0.1:8000](http://127.0.0.1:8000) in your web browser.

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

## 🧩 Model Context Protocol (MCP) Server Usage

Run the local Model Context Protocol server over standard I/O for integration with VS Code Copilot or Cursor:

```powershell
python mcp_server.py
```

Exposed MCP tools:
- `get_employee_profile(identifier)`
- `get_leave_balance(emp_id, leave_type)`
- `get_team_members(manager_id)`
- `get_department_stats()`
- `search_policy(query)`
- `draft_employee_email(recipient_emp_id, topic, action_requested, ...)`
