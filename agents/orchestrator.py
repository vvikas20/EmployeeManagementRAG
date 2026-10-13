import yaml
import time
import re
from typing import Dict, Any, List

from litellm_client import LiteLLMClient
from database import MSSQLDatabase
from vector_db import VectorDBManager
from agents.router_agent import RouterSupervisorAgent
from agents.policy_agent import PolicyRAGAgent
from agents.sql_agent import EmployeeSQLAgent
from agents.email_agent import EmailSynthesizerAgent
from agents.sql_tools import SQLToolExecutor
from agents.tool_agent import ToolCallingAgent
from agents.memory_agent import MemoryAgent
from agents.text_to_sql_agent import TextToSQLAgent


class MultiAgentOrchestrator:
    """
    Central Multi-Agent Orchestrator managing the execution pipeline.

    Feature flags (read from config.yaml → features):
      use_tool_calling  (Phase 1): When True, EMPLOYEE_SQL_QUERY / COMPANY_STATS_QUERY /
                                   HYBRID_POLICY_EMPLOYEE use the ToolCallingAgent loop
                                   instead of the fixed EmployeeSQLAgent functions.
                                   EMAIL_ACTION and POLICY_QUERY always use the original path.
    """

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

        features = cfg.get("features", {})
        self.use_tool_calling = features.get("use_tool_calling", False)
        self.use_conversation_context = features.get("use_conversation_context", False)
        self.use_text_to_sql = features.get("use_text_to_sql", False)

        self.llm         = LiteLLMClient(config_path)
        self.db          = MSSQLDatabase(config_path)
        self.vector_db   = VectorDBManager(config_path)

        self.router       = RouterSupervisorAgent(self.llm)
        self.policy_agent = PolicyRAGAgent(self.vector_db, self.llm)
        self.sql_agent    = EmployeeSQLAgent(self.db)
        self.email_agent  = EmailSynthesizerAgent(self.llm)

        # Phase 1, 2, & 4 components
        self.tool_executor = SQLToolExecutor(self.db)
        self.tool_agent    = ToolCallingAgent(self.llm, self.tool_executor)
        self.memory_agent  = MemoryAgent(self.llm, self.db)
        self.text_to_sql_agent = TextToSQLAgent(self.llm, self.db)

    # ─────────────────────────────────────────────────────────────────────────
    # Public entry point
    # ─────────────────────────────────────────────────────────────────────────

    def process_query(
        self, 
        user_query: str, 
        session_id: str = "default-session", 
        user_id: str = "anonymous", 
        role: str = "employee"
    ) -> Dict[str, Any]:
        start_time  = time.time()
        trace_steps: List[Dict[str, Any]] = []

        # Update caller identity in memory state
        self.memory_agent.update_session_state(session_id, {"caller_identity": user_id, "caller_role": role})

        # ── Step 0: Query Rewriting (Phase 2) ─────────────────────────────────
        standalone_query = user_query
        if self.use_conversation_context:
            trace_steps.append({
                "step": "Memory Agent Rewriting",
                "status": "Resolving pronouns and references using chat history...",
                "agent": "MemoryAgent"
            })
            rewrite_res = self.memory_agent.rewrite_query(session_id, user_query)
            
            if rewrite_res.get("needs_clarification"):
                trace_steps.append({
                    "step": "Clarification Required",
                    "status": "Query is ambiguous based on recent context.",
                    "details": rewrite_res["clarification_question"]
                })
                elapsed = round(time.time() - start_time, 2)
                return {
                    "query": user_query,
                    "intent": "CLARIFICATION",
                    "answer": rewrite_res["clarification_question"],
                    "email_draft": None,
                    "employee_data": None,
                    "policy_chunks": [],
                    "trace_steps": trace_steps,
                    "execution_time_seconds": elapsed,
                    "tool_calling_active": self.use_tool_calling,
                }

            standalone_query = rewrite_res.get("standalone_query", user_query)
            if standalone_query != user_query:
                trace_steps.append({
                    "step": "Query Rewritten",
                    "status": "Rewrote query to be fully standalone.",
                    "details": standalone_query
                })

        # ── Step 1: Supervisor Intent Routing ─────────────────────────────────
        trace_steps.append({
            "step":   "Supervisor Router Agent",
            "status": "Analyzing user intent and extracting entities...",
            "agent":  "RouterSupervisorAgent",
        })

        route   = self.router.route_query(standalone_query)
        intent  = route["intent"]
        emp_code = route.get("extracted_employee_code")

        trace_steps.append({
            "step":    "Intent Classification Result",
            "status":  f"Intent: {intent} | Target Employee: {emp_code or 'None'}",
            "details": route["reasoning"],
        })

        # Shared accumulators
        emp_data             = None
        policy_context_text  = ""
        retrieved_chunks: List[Dict[str, Any]] = []
        final_answer         = ""
        email_draft          = None

        # ── Branch: tool calling vs. classic ─────────────────────────────────
        if self.use_tool_calling and intent in (
            "EMPLOYEE_SQL_QUERY", "COMPANY_STATS_QUERY", "HYBRID_POLICY_EMPLOYEE"
        ):
            final_answer, trace_steps = self._run_with_tools(
                standalone_query, intent, emp_code, trace_steps
            )

        elif self.use_text_to_sql and intent == "CUSTOM_SQL_QUERY":
            trace_steps.append({
                "step":   "Text-to-SQL Agent",
                "status": "Generating and validating ad-hoc SQL query...",
                "agent":  "TextToSQLAgent",
            })
            sql_res = self.text_to_sql_agent.generate_and_execute(standalone_query)
            if sql_res["success"]:
                trace_steps.append({
                    "step": "SQL Executed Successfully",
                    "status": "Read-only query executed.",
                    "details": sql_res["sql"]
                })
                prompt = f"User Question: {standalone_query}\n\nSQL Results:\n{sql_res['results']}\n\nProvide a clear, natural language summary of the results."
                final_answer = self.llm.generate_completion(
                    prompt=prompt,
                    system_prompt="You are a data analyst assistant. Summarize the DB results concisely."
                )
            else:
                trace_steps.append({
                    "step": "SQL Execution Failed",
                    "status": "Validation or execution error.",
                    "details": sql_res.get("error", "Unknown error")
                })
                final_answer = f"I couldn't safely run that query. Error: {sql_res.get('error')}"

        else:
            # ── Classic (original) path ───────────────────────────────────────

            # Step 2: Company Stats OR Employee SQL Lookup
            if intent == "COMPANY_STATS_QUERY":
                trace_steps.append({
                    "step":   "Company Stats Agent",
                    "status": "Querying database for company-wide headcount and department metrics...",
                    "agent":  "EmployeeSQLAgent",
                })
                final_answer = self.sql_agent.get_company_overview()
                trace_steps.append({
                    "step":    "Company Stats Retrieved",
                    "status":  "Retrieved total employee count and department breakdown from database.",
                    "details": final_answer[:150] + "...",
                })

            elif emp_code or intent in ("EMPLOYEE_SQL_QUERY", "HYBRID_POLICY_EMPLOYEE", "EMAIL_ACTION"):
                target_id = emp_code if emp_code else "2212"
                trace_steps.append({
                    "step":   "Employee DB Agent",
                    "status": f"Executing SQL query for employee '{target_id}' in database...",
                    "agent":  "EmployeeSQLAgent",
                })
                emp_data = self.sql_agent.get_employee_data(target_id)
                if emp_data:
                    emp_summary = self.sql_agent.format_employee_summary(emp_data)
                    trace_steps.append({
                        "step":    "SQL Data Retrieved",
                        "status":  f"Found Employee: {emp_data.get('FirstName')} {emp_data.get('LastName')} ({emp_data.get('EmployeeNumber')})",
                        "details": emp_summary,
                    })

            # Step 3: Policy Vector Search if applicable
            if intent in ("POLICY_QUERY", "HYBRID_POLICY_EMPLOYEE", "EMAIL_ACTION"):
                trace_steps.append({
                    "step":   "Policy RAG Vector Agent",
                    "status": "Generating embeddings & performing similarity search in ChromaDB vector store...",
                    "agent":  "PolicyRAGAgent",
                })
                retrieved_chunks    = self.policy_agent.retrieve_context(standalone_query, top_k=3)
                policy_context_text = "\n\n".join([c["text"] for c in retrieved_chunks])
                trace_steps.append({
                    "step":    "Vector Chunks Retrieved",
                    "status":  f"Retrieved {len(retrieved_chunks)} relevant policy chunks from ChromaDB.",
                    "details": [f"Similarity: {c['similarity']} | {c['text'][:120]}..." for c in retrieved_chunks],
                })

            # Step 4: Synthesis
            if intent == "COMPANY_STATS_QUERY":
                pass  # Already set above

            elif intent == "EMAIL_ACTION":
                trace_steps.append({
                    "step":   "Email Synthesizer Agent",
                    "status": "Generating formal corporate email draft with Manager in CC...",
                    "agent":  "EmailSynthesizerAgent",
                })
                email_draft  = self.email_agent.generate_email(
                    user_prompt=standalone_query,
                    emp_data=emp_data,
                    policy_context=policy_context_text,
                )
                final_answer = (
                    f"Generated email draft for **{emp_data.get('FirstName', 'Employee')} "
                    f"{emp_data.get('LastName', '')}** keeping manager "
                    f"**{emp_data.get('ManagerName', 'Line Manager')}** in CC."
                )

            elif intent == "HYBRID_POLICY_EMPLOYEE":
                trace_steps.append({
                    "step":   "Multi-Agent Synthesis",
                    "status": "Fusing Database leave balances with ChromaDB Policy rules via LiteLLM...",
                    "agent":  "SupervisorSynthesizer",
                })
                emp_summary = (
                    self.sql_agent.format_employee_summary(emp_data)
                    if emp_data
                    else f"No employee record found for identifier '{emp_code or 'unknown'}' in the database."
                )
                synthesis_prompt = (
                    f"User Question: {standalone_query}\n\n"
                    f"Employee DB Profile & Leave Balances:\n{emp_summary}\n\n"
                    f"Company Policy Entitlements & Rules:\n{policy_context_text}\n\n"
                    f"Answer the user's question directly. Explicitly state the exact number of leaves "
                    f"employee {emp_code or '2212'} can apply for based on their current leave "
                    f"balances and company policy limits."
                )
                final_answer = self.llm.generate_completion(
                    prompt=synthesis_prompt,
                    system_prompt="You are a Lead Corporate HR & Operations AI Assistant.",
                    use_fast_model=False,
                )

            elif intent == "EMPLOYEE_SQL_QUERY":
                if emp_data:
                    final_answer = self.sql_agent.format_employee_summary(emp_data)
                else:
                    searched_id  = emp_code or "the requested identifier"
                    final_answer = (
                        f"No employee record was found for **{searched_id}** in the database. "
                        f"Please verify the employee code and try again, or check if the employee is active."
                    )

            else:  # POLICY_QUERY
                policy_res   = self.policy_agent.answer_policy_query(standalone_query, retrieved_chunks)
                final_answer = policy_res["answer"]

        # ── Step 5: Update session memory state ───────────────────────────────
        tracked_code = emp_code
        if not tracked_code and emp_data:
            tracked_code = emp_data.get("EmployeeNumber")
        if not tracked_code and final_answer:
            emp_match = re.search(r'\b(EMP\d{3,5})\b', final_answer)
            if emp_match:
                tracked_code = emp_match.group(1)

        if self.use_conversation_context and tracked_code:
            self.memory_agent.update_session_state(session_id, {
                "last_employee_code": tracked_code,
                "last_intent": intent
            })

        # ── Completion trace ──────────────────────────────────────────────────
        elapsed = round(time.time() - start_time, 2)
        trace_steps.append({
            "step":    "Execution Complete",
            "status":  f"Multi-Agent Workflow completed in {elapsed}s",
            "details": f"Total execution time: {elapsed} seconds | Tool calling: {self.use_tool_calling}",
        })

        return {
            "query":                  user_query,
            "intent":                 intent,
            "answer":                 final_answer,
            "email_draft":            email_draft,
            "employee_data":          emp_data,
            "policy_chunks":          retrieved_chunks,
            "trace_steps":            trace_steps,
            "execution_time_seconds": elapsed,
            "tool_calling_active":    self.use_tool_calling,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 1: Tool-calling execution path
    # ─────────────────────────────────────────────────────────────────────────

    def _run_with_tools(
        self,
        user_query: str,
        intent: str,
        emp_code: str | None,
        trace_steps: List[Dict[str, Any]],
    ) -> tuple[str, List[Dict[str, Any]]]:
        """
        Execute the ToolCallingAgent for SQL-related intents.
        For HYBRID_POLICY_EMPLOYEE, retrieves policy context first and passes it
        to the tool agent so the LLM can synthesize both data sources.
        Returns (final_answer, updated_trace_steps).
        """
        policy_context_text = ""

        # Fetch policy context for hybrid queries before entering the tool loop
        if intent == "HYBRID_POLICY_EMPLOYEE":
            trace_steps.append({
                "step":   "Policy RAG Vector Agent (pre-tool)",
                "status": "Retrieving policy context before tool-calling synthesis...",
                "agent":  "PolicyRAGAgent",
            })
            chunks = self.policy_agent.retrieve_context(user_query, top_k=3)
            policy_context_text = "\n\n".join([c["text"] for c in chunks])
            trace_steps.append({
                "step":    "Policy Context Ready",
                "status":  f"Retrieved {len(chunks)} policy chunks for hybrid synthesis.",
                "details": [f"Similarity: {c['similarity']} | {c['text'][:100]}..." for c in chunks],
            })

        trace_steps.append({
            "step":   "Tool-Calling Agent Loop",
            "status": f"Entering tool-calling loop for intent '{intent}'...",
            "agent":  "ToolCallingAgent",
        })

        result = self.tool_agent.run(
            user_query=user_query,
            policy_context=policy_context_text,
        )

        # Merge tool agent trace steps into the orchestrator trace
        trace_steps.extend(result.get("trace_steps", []))

        tool_summary = result.get("tool_calls_made", [])
        trace_steps.append({
            "step":    "Tool-Calling Agent Complete",
            "status":  (
                f"Completed in {result.get('iterations', '?')} iteration(s). "
                f"Tools used: {len(tool_summary)}"
            ),
            "details": [t["tool"] + (" ✓" if t["success"] else " ✗") for t in tool_summary],
        })

        return result["answer"], trace_steps

    def process_query_stream(
        self, 
        user_query: str, 
        session_id: str = "default-session", 
        user_id: str = "anonymous", 
        role: str = "employee"
    ):
        """Streaming generator equivalent of process_query."""
        start_time  = time.time()
        trace_steps: List[Dict[str, Any]] = []

        def yield_step(step_name: str, status: str, agent: str = None, details: Any = None):
            d = {"step": step_name, "status": status}
            if agent: d["agent"] = agent
            if details: d["details"] = details
            trace_steps.append(d)
            return {"type": "step", **d}

        self.memory_agent.update_session_state(session_id, {"caller_identity": user_id, "caller_role": role})

        standalone_query = user_query
        if self.use_conversation_context:
            yield yield_step("Memory Agent Rewriting", "Resolving pronouns and references using chat history...", "MemoryAgent")
            rewrite_res = self.memory_agent.rewrite_query(session_id, user_query)
            
            if rewrite_res.get("needs_clarification"):
                q = rewrite_res["clarification_question"]
                yield yield_step("Clarification Required", "Query is ambiguous based on recent context.", details=q)
                yield {"type": "complete", "result": {
                    "query": user_query, "intent": "CLARIFICATION", "answer": q, "email_draft": None,
                    "employee_data": None, "policy_chunks": [], "trace_steps": trace_steps,
                    "execution_time_seconds": round(time.time() - start_time, 2), "tool_calling_active": self.use_tool_calling
                }}
                return

            standalone_query = rewrite_res.get("standalone_query", user_query)
            if standalone_query != user_query:
                yield yield_step("Query Rewritten", "Rewrote query to be fully standalone.", details=standalone_query)

        yield yield_step("Supervisor Router Agent", "Analyzing user intent and extracting entities...", "RouterSupervisorAgent")
        route   = self.router.route_query(standalone_query)
        intent  = route["intent"]
        emp_code = route.get("extracted_employee_code")
        yield yield_step("Intent Classification Result", f"Intent: {intent} | Target Employee: {emp_code or 'None'}", details=route["reasoning"])

        emp_data = None
        policy_context_text = ""
        retrieved_chunks = []
        email_draft = None

        if self.use_tool_calling and intent in ("EMPLOYEE_SQL_QUERY", "COMPANY_STATS_QUERY", "HYBRID_POLICY_EMPLOYEE"):
            # Fetch policy context for hybrid before entering tool loop
            if intent == "HYBRID_POLICY_EMPLOYEE":
                yield yield_step("Policy RAG Vector Agent (pre-tool)", "Retrieving policy context before tool-calling synthesis...", "PolicyRAGAgent")
                retrieved_chunks = self.policy_agent.retrieve_context(standalone_query, top_k=3)
                policy_context_text = "\n\n".join([c["text"] for c in retrieved_chunks])
                yield yield_step("Policy Context Ready", f"Retrieved {len(retrieved_chunks)} policy chunks for hybrid synthesis.")

            yield yield_step("Tool-Calling Agent Loop", f"Entering tool-calling loop for intent '{intent}'...", "ToolCallingAgent")
            
            # Stream from tool agent
            final_answer = ""
            for evt in self.tool_agent.run_stream(standalone_query, policy_context=policy_context_text):
                if evt["type"] == "step":
                    trace_steps.append({"step": evt["step"], "status": evt["status"]})
                    yield evt
                elif evt["type"] == "token":
                    final_answer += evt["content"]
                    yield evt
                elif evt["type"] == "complete":
                    trace_steps.extend(evt["result"].get("trace_steps", []))
                    break
        elif self.use_text_to_sql and intent == "CUSTOM_SQL_QUERY":
            yield yield_step("Text-to-SQL Agent", "Generating and validating ad-hoc SQL query...", "TextToSQLAgent")
            sql_res = self.text_to_sql_agent.generate_and_execute(standalone_query)
            if sql_res["success"]:
                yield yield_step("SQL Executed Successfully", "Read-only query executed.", details=sql_res["sql"])
                prompt = f"User Question: {standalone_query}\n\nSQL Results:\n{sql_res['results']}\n\nProvide a clear, natural language summary of the results."
                final_answer = ""
                for chunk in self.llm.generate_completion_stream(prompt, system_prompt="You are a data analyst assistant. Summarize the DB results concisely."):
                    final_answer += chunk
                    yield {"type": "token", "content": chunk}
            else:
                yield yield_step("SQL Execution Failed", "Validation or execution error.", details=sql_res.get("error"))
                final_answer = f"I couldn't safely run that query. Error: {sql_res.get('error')}"
                yield {"type": "token", "content": final_answer}
        else:
            # Classic execution path, we just stream the final generation part
            if intent == "COMPANY_STATS_QUERY":
                yield yield_step("Company Stats Agent", "Querying database...", "EmployeeSQLAgent")
                final_answer = self.sql_agent.get_company_overview()
                yield yield_step("Company Stats Retrieved", "Retrieved total employee count.", details=final_answer[:150]+"...")
                yield {"type": "token", "content": final_answer}
            elif emp_code or intent in ("EMPLOYEE_SQL_QUERY", "HYBRID_POLICY_EMPLOYEE", "EMAIL_ACTION"):
                target_id = emp_code if emp_code else "2212"
                yield yield_step("Employee DB Agent", f"Executing SQL query for employee '{target_id}'...", "EmployeeSQLAgent")
                emp_data = self.sql_agent.get_employee_data(target_id)
                if emp_data:
                    yield yield_step("SQL Data Retrieved", f"Found Employee: {emp_data.get('FirstName')}", details=self.sql_agent.format_employee_summary(emp_data))

            if intent in ("POLICY_QUERY", "HYBRID_POLICY_EMPLOYEE", "EMAIL_ACTION"):
                yield yield_step("Policy RAG Vector Agent", "Generating embeddings...", "PolicyRAGAgent")
                retrieved_chunks = self.policy_agent.retrieve_context(standalone_query, top_k=3)
                policy_context_text = "\n\n".join([c["text"] for c in retrieved_chunks])
                yield yield_step("Vector Chunks Retrieved", f"Retrieved {len(retrieved_chunks)} relevant policy chunks.")

            if intent == "COMPANY_STATS_QUERY":
                pass
            elif intent == "EMAIL_ACTION":
                yield yield_step("Email Synthesizer Agent", "Generating formal corporate email draft...", "EmailSynthesizerAgent")
                email_draft = self.email_agent.generate_email(standalone_query, emp_data, policy_context_text)
                final_answer = f"Generated email draft for {emp_data.get('FirstName', 'Employee')}."
                yield {"type": "token", "content": final_answer}
            elif intent == "HYBRID_POLICY_EMPLOYEE":
                yield yield_step("Multi-Agent Synthesis", "Fusing DB and Policy...", "SupervisorSynthesizer")
                emp_summary = self.sql_agent.format_employee_summary(emp_data) if emp_data else f"No record found for '{emp_code}'."
                prompt = f"User Question: {standalone_query}\n\nEmployee:\n{emp_summary}\n\nPolicy:\n{policy_context_text}\n\nAnswer directly."
                final_answer = ""
                for chunk in self.llm.generate_completion_stream(prompt, system_prompt="You are a Lead Corporate HR & Operations AI Assistant."):
                    final_answer += chunk
                    yield {"type": "token", "content": chunk}
            elif intent == "EMPLOYEE_SQL_QUERY":
                if emp_data:
                    final_answer = self.sql_agent.format_employee_summary(emp_data)
                else:
                    final_answer = f"No employee record was found for {emp_code}."
                yield {"type": "token", "content": final_answer}
            else:
                context_text = "\n\n".join([f"--- Chunk ---\n{c['text']}" for c in retrieved_chunks])
                prompt = f"User Question: {standalone_query}\n\nContext:\n{context_text}\n\nProvide answer."
                final_answer = ""
                for chunk in self.llm.generate_completion_stream(prompt, system_prompt="You are an expert HR Policy Assistant. Answer accurately using context."):
                    final_answer += chunk
                    yield {"type": "token", "content": chunk}

        tracked_code = emp_code
        if not tracked_code and emp_data:
            tracked_code = emp_data.get("EmployeeNumber")
        if not tracked_code and final_answer:
            emp_match = re.search(r'\b(EMP\d{3,5})\b', final_answer)
            if emp_match:
                tracked_code = emp_match.group(1)

        if self.use_conversation_context and tracked_code:
            self.memory_agent.update_session_state(session_id, {"last_employee_code": tracked_code, "last_intent": intent})

        elapsed = round(time.time() - start_time, 2)
        yield yield_step("Execution Complete", f"Completed in {elapsed}s")

        yield {"type": "complete", "result": {
            "query": user_query, "intent": intent, "answer": final_answer, "email_draft": email_draft,
            "employee_data": emp_data, "policy_chunks": retrieved_chunks, "trace_steps": trace_steps,
            "execution_time_seconds": elapsed, "tool_calling_active": self.use_tool_calling
        }}
