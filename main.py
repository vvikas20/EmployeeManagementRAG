import os
import json
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import yaml

from database import DatabaseManager
from vector_db import VectorDBManager
from agents.orchestrator import MultiAgentOrchestrator

app = FastAPI(
    title="Enterprise Employee RAG & Multi-Agent System",
    description="SQLite + ChromaDB + LiteLLM + FastAPI Multi-Agent RAG System",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Core Services
orchestrator = MultiAgentOrchestrator()
db = DatabaseManager()
vector_db = VectorDBManager()

# Ingest default sample policy on startup if collection is empty or outdated
@app.on_event("startup")
def startup_event():
    print("Initializing Enterprise Employee RAG System...")
    stats = vector_db.get_collection_stats()
    sample_path = "sample_policy.md"
    if stats["total_chunks"] < 15 and os.path.exists(sample_path):
        with open(sample_path, "r", encoding="utf-8") as f:
            text = f.read()
        res = vector_db.ingest_document("Company_Leave_Policy_2026.md", text, category="Leave & Benefits Policy")
        print(f"Pre-loaded comprehensive sample policy into ChromaDB: {res}")

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = "default-session"
    user_id: Optional[str] = "anonymous"
    role: Optional[str] = "employee"

@app.get("/api/health")
def get_health():
    db_ok = db.test_connection()
    vdb_stats = vector_db.get_collection_stats()
    return {
        "status": "online" if db_ok else "degraded",
        "db_connected": db_ok,
        "mssql_connected": db_ok,
        "db_provider": "sqlite",
        "database": db.database,
        "vector_store": vdb_stats,
        "llm_provider": "LiteLLM"
    }

@app.post("/api/agent/query")
def process_agent_query(req: QueryRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query prompt cannot be empty.")
    
    try:
        result = orchestrator.process_query(
            user_query=req.query.strip(),
            session_id=req.session_id,
            user_id=req.user_id,
            role=req.role
        )
        db.log_chat_message(req.session_id, "user", req.query.strip())
        metadata = {"trace_steps": result.get("trace_steps", [])}
        if result.get("email_draft"):
            metadata["email_draft"] = result["email_draft"]
        db.log_chat_message(req.session_id, "assistant", result["answer"], json.dumps(jsonable_encoder(metadata)))
        return result
    except Exception as e:
        print(f"Error processing agent query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

from fastapi.responses import StreamingResponse

@app.post("/api/agent/query/stream")
def stream_agent_query(req: QueryRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query prompt cannot be empty.")
    
    def generate():
        db.log_chat_message(req.session_id, "user", req.query.strip())
        final_answer = ""
        trace_steps = []
        email_draft = None
        
        try:
            for event in orchestrator.process_query_stream(
                user_query=req.query.strip(),
                session_id=req.session_id,
                user_id=req.user_id,
                role=req.role
            ):
                if event["type"] == "token":
                    final_answer += event["content"]
                elif event["type"] == "complete":
                    trace_steps = event["result"].get("trace_steps", [])
                    email_draft = event["result"].get("email_draft")
                    
                # Yield SSE format
                yield f"data: {json.dumps(jsonable_encoder(event))}\n\n"
                
            # Log to DB after stream finishes
            metadata = {"trace_steps": trace_steps}
            if email_draft:
                metadata["email_draft"] = email_draft
            db.log_chat_message(req.session_id, "assistant", final_answer, json.dumps(jsonable_encoder(metadata)))
            
        except Exception as e:
            print(f"Error processing stream: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            
    return StreamingResponse(generate(), media_type="text/event-stream")

@app.post("/api/policy/upload")
async def upload_policy(
    file: UploadFile = File(...),
    category: str = Form("Corporate Policy")
):
    try:
        content_bytes = await file.read()
        filename = file.filename
        
        if filename.endswith(".pdf"):
            import pypdf
            import io
            pdf_reader = pypdf.PdfReader(io.BytesIO(content_bytes))
            text = "\n".join([page.extract_text() for page in pdf_reader.pages if page.extract_text()])
        else:
            text = content_bytes.decode("utf-8", errors="ignore")

        if not text.strip():
            raise HTTPException(status_code=400, detail="Could not extract readable text from uploaded file.")

        result = vector_db.ingest_document(filename, text, category=category)
        return {
            "message": "Policy document successfully ingested and vectorized.",
            "details": result,
            "vector_stats": vector_db.get_collection_stats()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/api/policy/ingest-sample")
def ingest_sample():
    sample_path = "sample_policy.md"
    if not os.path.exists(sample_path):
        raise HTTPException(status_code=404, detail="sample_policy.md not found")
    
    with open(sample_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    res = vector_db.ingest_document("Company_Leave_Policy_2026.md", text, category="Leave & Benefits Policy")
    return {"message": "Sample policy ingested", "details": res}

@app.get("/api/employees")
def list_employees(limit: int = 50):
    return db.list_all_employees(limit=limit)

@app.get("/api/employee/{emp_code}")
def get_employee_details(emp_code: str):
    emp = db.get_employee(emp_code)
    if not emp:
        raise HTTPException(status_code=404, detail=f"Employee '{emp_code}' not found.")
    return emp

@app.get("/api/vector-stats")
def get_vector_stats():
    return vector_db.get_collection_stats()

# Mount Static UI Files
os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
