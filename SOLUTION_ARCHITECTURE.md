# 🏗️ Technical Solution Architecture: Enterprise Employee RAG & Multi-Agent System

## 1. Executive Overview

The **Enterprise Employee RAG & Multi-Agent System** is a production-grade artificial intelligence solution designed to combine **structured enterprise relational data (SQLite)** with **unstructured knowledge retrieval (ChromaDB Vector RAG)** and **automated business action execution (Email Generation)**.

The architecture prioritizes zero-dependency portability and high performance:
- Relational data is managed via an embedded **SQLite** engine, auto-seeded with 28 realistic enterprise employees across 8 departments, 5 office locations, multi-tier management reporting, and 7 leave types.
- Policy knowledge is indexed into an embedded **ChromaDB** vector store using semantic embeddings (`mxbai-embed-large`).
- LLM inference is managed through **LiteLLM**, enabling effortless switching between cloud providers (**Groq** for ultra-fast response times, **Google AI Studio Gemini**) and offline local execution (**Ollama**).
- The solution is containerized via **Docker & Docker Compose** with persistent host volumes for data and vector storage.
- The web interface is engineered as a **minimalist, clean enterprise SPA** with real-time SSE streaming and a live execution trace visualizer.

---

## 2. Technical Stack & Infrastructure

| Subsystem | Technology | Purpose & Role |
| :--- | :--- | :--- |
| **Backend API** | **Python 3.10+ / FastAPI** | Asynchronous REST endpoints, static UI serving, Server-Sent Events (SSE) streaming |
| **Relational Database** | **SQLite (`sqlite3`)** | Self-contained relational store (`./data/employee_management.db`) with auto-seeding & safe schema view |
| **Vector Database** | **ChromaDB (Persistent)** | Embedded vector store (`./vector_store`) indexing the 10-section Corporate Policy manual (42 chunks) |
| **LLM Gateway** | **LiteLLM (`config.yaml`)** | Unified model client supporting selectable profiles (`groq`, `hosted`, `ollama`) with structured JSON parsing |
| **Cloud LLM (Primary)** | **Groq API** | High-throughput inference (`openai/gpt-oss-120b` for reasoning, `openai/gpt-oss-20b` for routing) |
| **Cloud LLM (Alternative)** | **Google Gemini** | Hosted profile using `gemini-3.8-flash` / `gemini-3.5-flash-lite` |
| **Local AI Services** | **Ollama** | Local profile running `llama3.1`, `qwen2.5:1.5b`, and `mxbai-embed-large` embeddings |
| **Containerization** | **Docker & Docker Compose** | Portable microservice packaging with volume mounts for database and vector store |
| **Frontend UI** | **HTML5 / Vanilla CSS / JS** | Minimalist SaaS aesthetic (slate/zinc palette), Suggested query chips, SSE streaming, and Execution Trace Timeline |
| **Agent Protocol** | **Model Context Protocol (MCP)** | Stdio-based server (`mcp_server.py`) for VS Code Copilot and Claude Desktop tool invocation |

---

## 3. High-Level System Architecture Diagram

```mermaid
flowchart TB
    subgraph ClientLayer["🖥️ Frontend & Client Layer"]
        UI["Minimal Web Dashboard (index.html)"]
        TraceUI["Live Agent Execution Trace Panel"]
        EmailUI["Corporate Email Preview & Copy Card"]
        MCPHost["VS Code Copilot / MCP Host"]
    end

    subgraph APILayer["⚡ FastAPI Gateway (main.py)"]
        HealthAPI["GET /api/health"]
        UploadAPI["POST /api/policy/upload"]
        QueryAPI["POST /api/agent/query/stream (SSE)"]
    end

    subgraph AgentLayer["🤖 Multi-Agent Orchestration Engine (agents/)"]
        Orchestrator["Multi-Agent Orchestrator (orchestrator.py)"]
        Memory["Memory & Context Agent (memory_agent.py)"]
        Supervisor["Supervisor Router Agent (router_agent.py)"]
        PolicyAgent["Policy RAG Agent (policy_agent.py)"]
        ToolAgent["Tool Calling Agent (tool_agent.py)"]
        TextToSQLAgent["Text-to-SQL Agent (text_to_sql_agent.py)"]
        EmailAgent["Email Synthesizer Agent (email_agent.py)"]
        SQLTools["Parameterized SQL Tool Executor (sql_tools.py)"]
        MCPServer["MCP Server (mcp_server.py)"]
    end

    subgraph DataLayer["💾 Persistence & AI Gateways"]
        SQLite[("SQLite Database\n(./data/employee_management.db)")]
        Chroma[("ChromaDB Vector Store\n(./vector_store)")]
        LiteLLM["LiteLLM Client Router\n(config.yaml)"]
        Groq["Groq API\n(gpt-oss-120b & 20b)"]
        Gemini["Google Gemini API\n(gemini-3.8-flash)"]
        Ollama["Local Ollama\n(mxbai-embed-large)"]
    end

    UI -->|Stream Query| QueryAPI
    UI -->|Policy Upload| UploadAPI
    QueryAPI --> Orchestrator
    MCPHost --> MCPServer
    MCPServer --> SQLTools
    SQLTools --> SQLite
    MCPServer --> Chroma
    MCPServer --> EmailAgent

    Orchestrator -->|1. Pronoun Resolution & Memory| Memory
    Memory -->|Conversation Context| SQLite
    Memory -->|Query Rewrite| LiteLLM

    Orchestrator -->|2. Route Intent| Supervisor
    Supervisor -->|Intent Classification| LiteLLM

    Orchestrator -->|TOOL_CALLING / SQL_LOOKUP| ToolAgent
    ToolAgent --> SQLTools
    SQLTools --> SQLite

    Orchestrator -->|CUSTOM_SQL_QUERY| TextToSQLAgent
    TextToSQLAgent -->|sqlglot AST Validation| SQLite

    Orchestrator -->|POLICY_SEARCH| PolicyAgent
    PolicyAgent -->|Cosine Search| Chroma
    PolicyAgent -->|Embeddings| Ollama

    Orchestrator -->|EMAIL_DRAFT| EmailAgent
    EmailAgent --> LiteLLM

    LiteLLM -->|active_profile: groq| Groq
    LiteLLM -->|active_profile: hosted| Gemini
    LiteLLM -->|active_profile: ollama| Ollama

    Orchestrator -->|Token-by-Token SSE Stream| QueryAPI
    QueryAPI --> UI
```

---

## 4. Subsystem & Component Details

### 4.1. Relational Database Layer (`database.py`)
Connects to an embedded SQLite database (`./data/employee_management.db`) using Python's standard `sqlite3` library.
- **Enterprise Schema:**
  - `Departments`: 8 business units (Executive, Engineering, Product, HR, Finance, Marketing, Customer Success, Sales).
  - `Designations`: 24 corporate roles from C-suite to senior specialists.
  - `Locations`: 5 office centers (Bengaluru HQ, Hyderabad, Mumbai, Pune, Remote/Hybrid).
  - `LeaveTypes`: 7 categories (`CL`, `EL`, `SL`, `ML`, `PL`, `BL`, `TL`).
  - `Employees`: 28 full employee profiles with manager hierarchies, corporate emails, hire dates, and salary figures.
  - `EmployeeLeaveBalances`: Available and used balances per employee.
  - `ChatSessions` & `ChatMessages`: Multi-turn conversational persistence.
- **Auto-Initialization & Seeding:** On startup, automatically creates tables and seeds full enterprise records if employee count is below threshold.
- **Safe Schema Introspection:** Exposes database schema via `PRAGMA table_info` while redacting sensitive columns like `Salary` for ad-hoc Text-to-SQL generation.

### 4.2. Vector Storage & Ingestion (`vector_db.py`)
Indexes company policies into an embedded persistent ChromaDB collection.
- **Document Chunking:** Recursive paragraph and sentence chunking (`chunk_size=500`, `chunk_overlap=50`).
- **Embeddings:** Generates vector representations via Ollama's `mxbai-embed-large` (1024 dimensions) with resilient timeout and fast fallback handling.
- **Policy Ingestion:** Automatically ingests the 10-section [`sample_policy.md`](./sample_policy.md) (42 chunks).

### 4.3. Multi-Agent Engine (`agents/`)
- **`memory_agent.py`:** Analyzes recent chat turns to resolve pronouns ("his manager", "her leave balance", "write an email to her") to active entity IDs (e.g. `EMP8000`) before routing, avoiding redundant clarification loops. Enforces ~1000 token budgeting on history context.
- **`router_agent.py`:** Employs the `fast-llm` model to classify incoming queries into six functional intents with strict JSON schemas:
  1. `COMPANY_STATS_QUERY`: Company-wide employee counts and department metrics.
  2. `EMAIL_ACTION`: Drafting formal corporate emails.
  3. `HYBRID_POLICY_EMPLOYEE`: Questions combining company policy rules with individual employee balances.
  4. `EMPLOYEE_SQL_QUERY`: Queries asking for a specific employee's profile or leave balances.
  5. `CUSTOM_SQL_QUERY`: Ad-hoc or complex database queries.
  6. `POLICY_QUERY`: General policy, benefits, and HR guidelines.
- **`tool_agent.py` & `sql_tools.py`:** Autonomous agent loop using function calling against safe, parameterized SQL primitives (`get_employee_profile`, `get_leave_balance`, `get_team_members`, `get_department_stats`).
- **`text_to_sql_agent.py`:** Generates arbitrary SQLite analytical queries guarded by `sqlglot` (enforcing `SELECT`-only statements, banning mutation keywords, and enforcing `LIMIT 50`).
- **`email_agent.py`:** Synthesizes professional corporate communications with recipient validation and manager CC lookups.
- **`orchestrator.py`:** State machine coordinating agent transitions and yielding real-time execution traces for frontend observability.

### 4.4. Model Context Protocol (MCP) Server (`mcp_server.py`)
Exposes the multi-agent tools over standard I/O (stdio) compliant with the open Model Context Protocol. Compatible with VS Code GitHub Copilot, Cursor, and Claude Desktop.

---

## 5. End-to-End Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    actor User as User / Web Client
    participant API as FastAPI (SSE)
    participant Orch as Orchestrator
    participant Mem as Memory Agent
    participant Router as Supervisor Router
    participant Tool as Tool Calling Agent
    participant TextSQL as Text-to-SQL Agent
    participant DB as SQLite Database
    participant RAG as Policy RAG Agent
    participant Chroma as ChromaDB Vector Store
    participant LLM as LiteLLM (Groq / Gemini)

    User->>API: POST /api/agent/query/stream {"query": "..."}
    API->>Orch: process_query_stream(user_query)
    
    Orch->>Mem: rewrite_query(user_query)
    Mem->>DB: Fetch recent session messages & state
    Mem->>LLM: Resolve pronouns & context
    LLM-->>Mem: standalone_query
    
    Orch->>Router: route_query(standalone_query)
    Router->>LLM: Intent Classification (JSON)
    LLM-->>Router: {"intent": "...", "confidence": 0.95}
    
    alt If Structured Employee Query (Tool Calling)
        Orch->>Tool: run_stream()
        Tool->>LLM: Reason & choose tool
        LLM-->>Tool: Tool Call: get_employee_profile(EMP8000)
        Tool->>DB: Parameterized SQLite Query
        DB-->>Tool: Employee record
    else If Analytical Ad-Hoc SQL
        Orch->>TextSQL: generate_and_execute()
        TextSQL->>LLM: Generate SQLite query
        TextSQL->>TextSQL: sqlglot AST Check (SELECT only, LIMIT 50)
        TextSQL->>DB: execute_read_only_query()
        DB-->>TextSQL: Result rows
    else If Policy Search Required
        Orch->>RAG: retrieve_context()
        RAG->>Chroma: Cosine similarity vector search
        Chroma-->>RAG: Top matching policy chunks
    end

    Orch->>LLM: Synthesize final response
    loop SSE Stream
        LLM-->>Orch: Yield token chunk
        Orch-->>API: yield {"type": "token", "content": "..."}
        API-->>User: Progressive Markdown rendering
    end

    Orch->>DB: log_chat_message()
    Orch-->>API: yield {"type": "complete", "trace": [...]}
    API-->>User: Complete response & execution trace
```

---

## 6. Interactive Debugging Architecture

The system supports granular step debugging across all agent execution stages via standard IDE debuggers:

```mermaid
flowchart LR
    A["Breakpoint 1:<br>memory_agent.py (L56)"] --> B["Breakpoint 2:<br>router_agent.py (L17)"]
    B --> C["Breakpoint 3:<br>orchestrator.py (L408)"]
    C --> D["Breakpoint 4:<br>sql_tools.py (L165)"]
    C --> E["Breakpoint 5:<br>policy_agent.py (L17)"]
    C --> F["Breakpoint 6:<br>email_agent.py (L12)"]
```

Developers can attach the Python debugger (`F5`) to inspect runtime variables (`session_id`, `emp_data`, `retrieved_chunks`, `tool_calls_made`) and step through decisions before responses stream to the client.

---

## 7. Security, Privacy & Safety Guardrails

1. **Local Data Sovereignty:** Relational data (`./data/employee_management.db`) and vector indices (`./vector_store/`) are stored locally on disk and never uploaded to external databases.
2. **Read-Only AST Validation:** All Text-to-SQL outputs pass through `sqlglot` AST parsing. Statements containing `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, or `ATTACH` are blocked before reaching SQLite.
3. **Sensitive Field Protection:** Ad-hoc SQL schemas redact sensitive financial columns like `Salary` by default.
4. **Parameterized Primitives:** Standard profile lookups and leave calculations use bound parameters (`?`), preventing SQL injection.
5. **Row Retrieval Caps:** Ad-hoc queries automatically enforce `LIMIT 50` to mitigate denial-of-service and memory exhaustion.
6. **Credential Hygiene:** Cloud LLM API keys (`GROQ_API_KEY`, `GEMINI_API_KEY`) reside exclusively in git-ignored `config.local.yaml`.
