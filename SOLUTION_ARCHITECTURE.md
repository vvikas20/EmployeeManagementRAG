# 🏗️ Technical Solution Architecture: Enterprise Employee RAG & Multi-Agent System

## 1. Executive Overview

The **Enterprise Employee RAG & Multi-Agent System** is a production-grade artificial intelligence solution designed to combine **structured enterprise relational data (SQLite)** with **unstructured knowledge retrieval (ChromaDB Vector RAG)** and **automated business action execution (Email Generation)**.

The architecture prioritizes zero-dependency portability and high performance:
- Relational data is managed via an embedded **SQLite** engine, auto-seeded with 28 realistic enterprise employees across 8 departments, 5 office locations, multi-tier management reporting, and 7 leave types.
- Policy knowledge is indexed into an embedded **ChromaDB** vector store using semantic embeddings.
- LLM inference is managed through **LiteLLM**, enabling effortless switching between cloud providers (**Groq** for ultra-fast response times, **Google AI Studio Gemini**) and offline local execution (**Ollama**).
- The solution is containerized via **Docker & Docker Compose** with persistent host volumes for data and vector storage.

---

## 2. Technical Stack & Infrastructure

| Subsystem | Technology | Purpose & Role |
| :--- | :--- | :--- |
| **Backend API** | **Python 3.10+ / FastAPI** | Asynchronous REST endpoints, static UI serving, Server-Sent Events (SSE) streaming |
| **Relational Database** | **SQLite (`sqlite3`)** | Self-contained relational store (`./data/employee_management.db`) with auto-seeding & safe schema view |
| **Vector Database** | **ChromaDB (Persistent)** | Embedded vector store (`./vector_store`) indexing the 10-section Corporate Policy manual |
| **LLM Gateway** | **LiteLLM (`config.yaml`)** | Unified model client supporting selectable profiles (`groq`, `gemini`, `ollama`) with structured JSON parsing |
| **Cloud LLM (Primary)** | **Groq API** | High-throughput inference (`openai/gpt-oss-120b` for reasoning, `openai/gpt-oss-20b` for routing) |
| **Cloud LLM (Alternative)** | **Google Gemini** | Hosted profile using `gemini-2.5-flash` / `gemini-2.5-flash-lite` |
| **Local AI Services** | **Ollama** | Local profile running `llama3.1`, `qwen2.5:1.5b`, and `mxbai-embed-large` embeddings |
| **Containerization** | **Docker & Docker Compose** | Portable microservice packaging with volume mounts for database and vector store |
| **Frontend UI** | **HTML5 / Vanilla CSS / JS** | Light theme responsive SPA with live Multi-Agent Execution Trace Viewer & Progressive Markdown Rendering |
| **Agent Protocol** | **Model Context Protocol (MCP)** | Stdio-based server (`mcp_server.py`) for VS Code Copilot and Claude Desktop tool invocation |

---

## 3. High-Level System Architecture Diagram

```mermaid
flowchart TB
    subgraph ClientLayer["🖥️ Frontend & Client Layer"]
        UI["Web Dashboard (index.html)"]
        TraceUI["Live Agent Trace Visualizer"]
        EmailUI["Email Action Studio"]
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
        Gemini["Google Gemini API\n(gemini-2.5-flash)"]
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
    LiteLLM -->|active_profile: gemini| Gemini
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
  - `Employees`: 28 full employee profiles with manager hierarchies, emails, hire dates, and salary figures.
  - `EmployeeLeaveBalances`: Available and used balances per employee.
  - `ChatSessions` & `ChatMessages`: Multi-turn conversational persistence.
- **Auto-Initialization & Seeding:** On startup, automatically creates tables and seeds full enterprise records if employee count is below threshold.
- **Safe Schema Introspection:** Exposes database schema via `PRAGMA table_info` while redacting sensitive columns like `Salary` for ad-hoc Text-to-SQL generation.

### 4.2. Vector Storage & Ingestion (`vector_db.py`)
Indexes company policies into an embedded persistent ChromaDB collection.
- **Document Chunking:** Recursive paragraph and sentence chunking (`chunk_size=500`, `chunk_overlap=50`).
- **Embeddings:** Generates vector representations via Ollama's `mxbai-embed-large` (1024 dimensions) with resilient timeout and fallback handling.
- **Policy Ingestion:** Automatically ingests the 10-section [`sample_policy.md`](file:///c:/D/Playground/Gen%20AI/EmployeeManagementRAGSystem/sample_policy.md) (42 chunks).

### 4.3. Multi-Agent Engine (`agents/`)
- **`memory_agent.py`:** Analyzes recent chat turns to resolve pronouns ("his manager", "her leave balance", "write him an email") to active entity IDs (e.g. `EMP4001`) before routing, avoiding redundant clarification loops.
- **`router_agent.py`:** Employs the `fast-llm` model to classify incoming queries into six functional intents with strict JSON schemas.
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
        LLM-->>Tool: Tool Call: get_employee_profile(EMP4001)
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

## 6. Multi-Agent Intent Routing Logic

```mermaid
graph TD
    Query[User Query] --> Router{Supervisor Router Agent}
    
    Router -->|COMPANY_STATS| ToolTrack["Tool Calling Agent<br>(get_department_stats)"]
    Router -->|EMPLOYEE_SQL| ToolTrack
    Router -->|HYBRID_POLICY_EMPLOYEE| HybridTrack["Multi-Agent Synthesis<br>(Tool Agent + Policy RAG)"]
    
    Router -->|CUSTOM_SQL_QUERY| AdHocTrack["Text-to-SQL Agent<br>(sqlglot Validated SQLite)"]
    Router -->|POLICY_QUERY| PolicyTrack["Policy RAG Agent<br>(ChromaDB Cosine Retrieval)"]
    Router -->|EMAIL_ACTION| EmailTrack["Email Synthesizer Agent<br>(Manager CC + Action Draft)"]
```

---

## 7. Security, Privacy & Safety Guardrails

1. **Local Data Sovereignty:** Relational data (`./data/employee_management.db`) and vector indices (`./vector_store/`) are stored locally on disk and never uploaded to external databases.
2. **Read-Only AST Validation:** All Text-to-SQL outputs pass through `sqlglot` AST parsing. Statements containing `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, or `ATTACH` are blocked before reaching SQLite.
3. **Sensitive Field Redaction:** The schema provided to the LLM for custom SQL explicitly omits sensitive financial columns such as `Salary`.
4. **Parameterized Primitives:** Standard profile lookups and leave calculations use bound parameters (`?`), preventing SQL injection.
5. **Row Retrieval Caps:** Ad-hoc queries automatically enforce `LIMIT 50` to mitigate denial-of-service and memory exhaustion.
6. **Credential Hygiene:** Cloud LLM API keys (`GROQ_API_KEY`, `GEMINI_API_KEY`) reside exclusively in git-ignored `config.local.yaml`.
