"""
agents/memory_agent.py
Phase 2 — Conversation context and query rewriting.

Manages in-memory session states, fetches chat history from MSSQL,
and uses fast-llm to rewrite queries to be standalone.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

import tiktoken

from database import MSSQLDatabase
from litellm_client import LiteLLMClient


class MemoryAgent:
    def __init__(self, llm: LiteLLMClient, db: MSSQLDatabase):
        self.llm = llm
        self.db = db
        # In-memory session state: session_id -> dict
        self.session_states: Dict[str, Dict[str, Any]] = {}
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def get_session_state(self, session_id: str) -> Dict[str, Any]:
        """Retrieve or initialize session state."""
        if session_id not in self.session_states:
            self.session_states[session_id] = {
                "last_employee_code": None,
                "last_intent": None,
                "last_topic": None,
                "caller_identity": "anonymous"
            }
        return self.session_states[session_id]

    def update_session_state(self, session_id: str, updates: Dict[str, Any]) -> None:
        """Update fields in the session state."""
        state = self.get_session_state(session_id)
        state.update(updates)

    def _count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken (cl100k_base)."""
        return len(self.tokenizer.encode(text))

    def _summarize_history(self, history: List[Dict[str, str]]) -> str:
        """Summarize older chat history using fast-llm."""
        if not history:
            return ""
        text = "\n".join([f"{m['role']}: {m['content']}" for m in history])
        prompt = f"Summarize the following chat history briefly to capture the context and main entities discussed:\n\n{text}"
        system = "You are a helpful assistant. Keep the summary extremely concise."
        return self.llm.generate_completion(prompt, system, use_fast_model=True)

    def rewrite_query(self, session_id: str, current_query: str) -> Dict[str, Any]:
        """
        Rewrite the query to be standalone based on conversation history and rules.
        Returns a dictionary with:
          - standalone_query
          - needs_clarification
          - clarification_question
        """
        state = self.get_session_state(session_id)
        
        # 1. Load history (fetch up to 50 to see if we cross the 10-turn threshold)
        full_history = self.db.get_chat_history(session_id, limit=50)
        
        summary = ""
        recent_history = full_history
        if len(full_history) > 10:
            older_turns = full_history[:-6]
            recent_history = full_history[-6:]
            summary = self._summarize_history(older_turns)

        # 2. Build context string
        context_str = ""
        if summary:
            context_str += f"[Older history summary: {summary}]\n\n"
        for m in recent_history:
            context_str += f"{m['role']}: {m['content']}\n"

        # Enforce ~1000 token limit on the context string
        while self._count_tokens(context_str) > 1000 and len(recent_history) > 2:
            recent_history = recent_history[1:]
            context_str = "\n".join([f"{m['role']}: {m['content']}" for m in recent_history])

        # 3. Rule-based reference resolution hints
        rule_hints = []
        last_emp = state.get("last_employee_code")

        # Fallback: scan recent history for employee code e.g. EMP4001 or EMP2212 if not in state
        if not last_emp and recent_history:
            for m in reversed(recent_history):
                m_code = re.search(r'\b(EMP\d{3,5})\b', m.get("content", ""))
                if m_code:
                    last_emp = m_code.group(1)
                    state["last_employee_code"] = last_emp
                    break

        q_lower = current_query.lower()

        if last_emp:
            emp = self.db.get_employee(last_emp)
            if emp:
                first = emp.get("FirstName", "")
                last = emp.get("LastName", "")
                emp_name = f"{first} {last}".strip() or emp.get("EmployeeNumber", f"employee {last_emp}")
            else:
                emp_name = f"employee {last_emp}"

            # "his manager", "her manager", etc.
            if any(phrase in q_lower for phrase in ["his manager", "her manager", "their manager", "the manager", "my manager"]):
                if emp and emp.get("ManagerId"):
                    mgr_id = emp["ManagerId"]
                    mgr = self.db.get_employee(str(mgr_id))
                    if mgr:
                        mgr_code = mgr.get("EmployeeNumber", str(mgr_id))
                        rule_hints.append(f"- 'manager' refers to manager employee code '{mgr_code}'")

            # Pronouns or action references
            if re.search(r'\b(he|his|him|she|her|they|their|them|the employee)\b', q_lower) or any(action in q_lower for action in ["email", "certificate", "leave", "profile", "salary", "balance"]):
                rule_hints.append(f"- Active employee currently in discussion: '{emp_name}' (Code: {last_emp})")
                rule_hints.append(f"- Pronouns (he, him, his, she, her, they, them, the employee) refer to '{emp_name}' ({last_emp})")

        # 4. LLM Rewriting
        system_prompt = (
            "You are an intelligent query rewriting assistant for an enterprise employee system.\n"
            "Rewrite the user's latest query to be fully standalone based on the chat history and rules.\n"
            "Guidelines:\n"
            "1. If a single employee was recently discussed or identified in the conversation, resolve pronouns (e.g. 'him', 'her', 'them', 'the employee') to that employee's name and employee code.\n"
            "2. Do NOT be pedantic about gender pronouns (e.g. if the user says 'write him an email' but the employee discussed is named Ananya, resolve it to Ananya Deshmukh unless another male employee was explicitly discussed).\n"
            "3. Only set 'needs_clarification' to true if multiple different employees are active in context with equal ambiguity and it is genuinely impossible to tell who is meant.\n"
            "4. If the user query is already standalone or clear, output it as is.\n"
            "Respond ONLY in JSON format:\n"
            "{\n"
            '  "standalone_query": "...",\n'
            '  "needs_clarification": false,\n'
            '  "clarification_question": null\n'
            "}"
        )

        hints_str = f"Rule-based Entity Hints:\n{chr(10).join(rule_hints)}\n\n" if rule_hints else ""
        prompt = (
            f"Chat History:\n{context_str}\n\n"
            f"{hints_str}"
            f"Current Query: {current_query}\n\n"
            f"Provide the JSON output:"
        )

        res = self.llm.generate_json_structured(prompt, system_prompt, use_fast_model=True)

        if res.get("error"):
            # Fallback on LLM failure
            return {
                "standalone_query": current_query,
                "needs_clarification": False,
                "clarification_question": None
            }

        return {
            "standalone_query": str(res.get("standalone_query", current_query)),
            "needs_clarification": bool(res.get("needs_clarification", False)),
            "clarification_question": str(res.get("clarification_question", "")) if res.get("clarification_question") else None
        }
