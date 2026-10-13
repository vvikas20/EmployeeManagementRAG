# 🏗️ Technical Architecture Report: Enterprise Employee RAG & Multi-Agent System

> [!NOTE]
> This document details the technical architecture, data flows, multi-agent orchestration, and component interactions of the **Enterprise Employee RAG System** powered by **SQLite**, **ChromaDB**, and **LiteLLM** (supporting Groq, Google Gemini, and local Ollama), fully deployable locally or via **Docker**.

---

## 1. High-Level Architecture Diagram

```mermaid
flowchart TB
    subgraph ClientLayer["🖥️ Frontend UI Layer (Light Mode SPA)"]
        UI["Web Workspace (index.html / styles.css)"]
        TraceUI["Live Agent Trace Visualizer"]
        EmailUI["Email Action Studio"]
        MCPHost["VS Code Copilot / MCP Host"]
    end

    subgraph APILayer["⚡ FastAPI Backend (main.py)"]
        HealthAPI["GET /api/health"]
        UploadAPI["POST /api/policy/upload"]
        QueryAPI["POST /api/agent/query/stream (SSE)"]
        EmpAPI["GET /api/employees"]
    end

    subgraph AgentLayer["🤖 Multi-Agent Orchestration Engine (agents/)"]
        Orchestrator["Multi-Agent Orchestrator (orchestrator.py)"]
        Memory["Memory & Context Agent (memory_agent.py)"]
        Supervisor["Supervisor Router Agent (router_agent.py)"]
        PolicyAgent["Policy RAG Agent (policy_agent.py)"]
        ToolAgent["Tool Calling Agent (tool_agent.py)"]
        SQLAgent["Employee SQL Agent (sql_agent.py)"]
        EmailAgent["Email Synthesizer Agent (email_agent.py)"]
        MCPServer["Local MCP Server (mcp_server.py, stdio)"]
    end

    subgraph DataLayer["💾 Enterprise Storage & Configurable AI Providers"]
        SQLite[("SQLite Database\n(./data/employee_management.db)")]
        Chroma[("ChromaDB Vector Store\n(./vector_store)")]
        LiteLLM["LiteLLM Client\n(selectable profiles in config.yaml)"]
        Groq["Groq API\n(gpt-oss-120b / gpt-oss-20b)"]
        Gemini["Google Gemini API\n(gemini-2.5-flash)"]
        Ollama["Local Ollama\n(mxbai-embed-large)"]
    end

    UI -->|Stream Query & Upload| QueryAPI
    QueryAPI --> Orchestrator
    MCPHost --> MCPServer
    MCPServer -->|Employee and leave tools| SQLite
    MCPServer -->|Policy search| Chroma
    MCPServer -->|Email draft tool| EmailAgent
    
    Orchestrator --> Memory
    Memory -->|Context & Pronoun Resolution| LiteLLM
    Memory -->|Session History| SQLite
    
    Orchestrator --> Supervisor
    Supervisor -->|Classify Intent & Extract Entities| Orchestrator
    
    Orchestrator -->|Company Metrics & Profile| ToolAgent
    ToolAgent --> SQLite
    
    Orchestrator -->|Policy Search| PolicyAgent
    PolicyAgent -->|Cosine Search| Chroma
    PolicyAgent -->|Embeddings| Ollama
    
    Orchestrator -->|Email Draft| EmailAgent
    EmailAgent --> LiteLLM
    
    Supervisor --> LiteLLM
    
    LiteLLM -->|active profile: groq| Groq
    LiteLLM -->|active profile: gemini| Gemini
    LiteLLM -->|active profile: ollama| Ollama
    
    Orchestrator -->|Synthesized Output + Traces| QueryAPI
    QueryAPI --> UI
```

---

## 2. Component Breakdown

| Subsystem | File Reference | Primary Function |
| :--- | :--- | :--- |
| **FastAPI Web Server** | [`main.py`](file:///c:/D/Playground/Gen%20AI/EmployeeManagementRAGSystem/main.py) | REST API endpoints (`/api/agent/query/stream`, `/api/policy/upload`, `/api/health`, `/api/employees`) and static SPA hosting. |
| **Relational Database** | [`database.py`](file:///c:/D/Playground/Gen%20AI/EmployeeManagementRAGSystem/database.py) | Self-contained SQLite database (`./data/employee_management.db`), auto-seeded with 28 employees, 8 departments, 24 designations, 7 leave types, and safe schema projection. |
| **Vector DB Store** | [`vector_db.py`](file:///c:/D/Playground/Gen%20AI/EmployeeManagementRAGSystem/vector_db.py) | ChromaDB persistent store indexing the 10-section Corporate Policy manual with local embedding generation and fallback handling. |
| **LLM Gateway** | [`litellm_client.py`](file:///c:/D/Playground/Gen%20AI/EmployeeManagementRAGSystem/litellm_client.py), [`config.yaml`](file:///c:/D/Playground/Gen%20AI/EmployeeManagementRAGSystem/config.yaml) | Unified interface supporting Groq (`openai/gpt-oss-120b`, `openai/gpt-oss-20b`), Google Gemini (`gemini-2.5-flash`), and local Ollama (`llama3.1`). |
| **Memory Agent** | [`agents/memory_agent.py`](file:///c:/D/Playground/Gen%20AI/EmployeeManagementRAGSystem/agents/memory_agent.py) | Context-aware pronoun and entity resolution across multi-turn sessions (e.g. mapping "write him an email" to `EMP4001`). |
| **Supervisor Router** | [`agents/router_agent.py`](file:///c:/D/Playground/Gen%20AI/EmployeeManagementRAGSystem/agents/router_agent.py) | Classifies query intent into `COMPANY_STATS_QUERY`, `POLICY_QUERY`, `EMPLOYEE_SQL_QUERY`, `HYBRID_POLICY_EMPLOYEE`, `CUSTOM_SQL_QUERY`, or `EMAIL_ACTION`. |
| **Tool Calling Agent** | [`agents/tool_agent.py`](file:///c:/D/Playground/Gen%20AI/EmployeeManagementRAGSystem/agents/tool_agent.py), [`agents/sql_tools.py`](file:///c:/D/Playground/Gen%20AI/EmployeeManagementRAGSystem/agents/sql_tools.py) | Executes parameterized queries (`get_employee_profile`, `get_leave_balance`, `get_team_members`, `get_department_stats`). |
| **Text-to-SQL Agent** | [`agents/text_to_sql_agent.py`](file:///c:/D/Playground/Gen%20AI/EmployeeManagementRAGSystem/agents/text_to_sql_agent.py) | Generates read-only SQLite analytical queries protected by `sqlglot` AST safety rules. |
| **Policy RAG Agent** | [`agents/policy_agent.py`](file:///c:/D/Playground/Gen%20AI/EmployeeManagementRAGSystem/agents/policy_agent.py) | Semantic search on ChromaDB policy vectors and RAG context synthesis. |
| **Email Agent** | [`agents/email_agent.py`](file:///c:/D/Playground/Gen%20AI/EmployeeManagementRAGSystem/agents/email_agent.py) | Crafts professional corporate emails with `To`, `CC` (Manager), `Subject`, and formatted `Body`. |
| **Multi-Agent Coordinator** | [`agents/orchestrator.py`](file:///c:/D/Playground/Gen%20AI/EmployeeManagementRAGSystem/agents/orchestrator.py) | Manages state machine transitions and streams real-time execution traces. |
| **MCP Server** | [`mcp_server.py`](file:///c:/D/Playground/Gen%20AI/EmployeeManagementRAGSystem/mcp_server.py) | Exposes multi-agent tools over standard I/O for VS Code Copilot and Claude Desktop. |

---

## 3. End-to-End Sequence Diagram

```mermaid
sequenceDiagram
    autonumber
    actor User as User / Web Client
    participant API as FastAPI (SSE)
    participant Orch as Orchestrator
    participant Mem as Memory Agent
    participant Router as Supervisor Router
    participant Tool as Tool Calling Agent
    participant DB as SQLite Database
    participant RAG as Policy RAG Agent
    participant Chroma as ChromaDB
    participant LLM as LiteLLM (Groq / Gemini)

    User->>API: POST /api/agent/query/stream {"query": "..."}
    API->>Orch: process_query_stream(user_query)

    Orch->>Mem: rewrite_query(user_query)
    Mem->>DB: Read conversation history & recent entities
    Mem->>LLM: Resolve pronouns to employee codes
    LLM-->>Mem: Rewritten standalone query
    
    Orch->>Router: route_query(standalone_query)
    Router->>LLM: JSON classification
    LLM-->>Router: {"intent": "EMPLOYEE_SQL_QUERY"}
    
    Orch->>Tool: run_stream()
    Tool->>LLM: Model chooses function call
    LLM-->>Tool: Call get_employee_profile("EMP4001")
    Tool->>DB: Parameterized SQLite lookup
    DB-->>Tool: Employee record
    
    Tool->>LLM: Synthesize final response
    loop SSE Stream
        LLM-->>Orch: Token chunk
        Orch-->>API: yield {"type": "token", "content": "..."}
        API-->>User: Stream to UI
    end

    Orch->>DB: log_chat_message()
    Orch-->>API: yield {"type": "complete"}
    API-->>User: Finalize message & trace display
```

---

## 4. Deployment & Infrastructure

- **Zero External Dependencies**: Operates with zero requirement for external SQL server instances or client drivers.
- **Dockerized Architecture**: Packaged as a single container via `Dockerfile` and `docker-compose.yml` with persistent storage for `./data` and `./vector_store`.
- **Health Verification**: `/api/health` exposes database status, vector count, and active LLM profile.
