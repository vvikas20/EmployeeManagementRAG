"""
tests/test_phase2.py
Phase 2 — pytest suite for memory_agent.py.

All tests mock litellm.completion so no real LLM or database is needed.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
import pytest

from agents.memory_agent import MemoryAgent

class StubDB:
    def __init__(self):
        self.history = []

    def get_employee(self, emp_id: str):
        if str(emp_id) in ("2212", "EMP2212"):
            return {"EmployeeNumber": "EMP2212", "ManagerId": 1001}
        if str(emp_id) in ("1001", "EMP1001"):
            return {"EmployeeNumber": "EMP1001"}
        return None

    def get_chat_history(self, session_id: str, limit: int = 50):
        return self.history[-limit:] if self.history else []

class StubLLM:
    def __init__(self):
        pass

    def generate_completion(self, prompt: str, system_prompt: str, use_fast_model: bool = False, temperature: float = 0.2):
        return "This is a summary of older turns."

    def generate_json_structured(self, prompt: str, system_prompt: str, use_fast_model: bool = False):
        if "needs_clarification" in prompt.lower() and "which one" in prompt.lower():
             return {
                "needs_clarification": True,
                "clarification_question": "Which employee are you referring to?"
            }
        
        # Simple mock logic based on query content
        if "his manager" in prompt.lower():
            return {"standalone_query": "What is EMP1001's email?", "needs_clarification": False}
        if "what is his leave" in prompt.lower():
            return {"standalone_query": "What is EMP2212's leave balance?", "needs_clarification": False}
        
        return {"standalone_query": "Mocked standalone query"}

class TestMemoryAgent:
    @pytest.fixture
    def agent(self):
        db = StubDB()
        llm = StubLLM()
        return MemoryAgent(llm, db)

    def test_no_history_standalone(self, agent):
        res = agent.rewrite_query("session_1", "What is Vikram's email?")
        assert res["needs_clarification"] is False
        # StubLLM returns this for unknown prompts
        assert res["standalone_query"] == "Mocked standalone query"

    def test_pronoun_resolution_rules(self, agent):
        # Emulate session state with last_employee_code
        agent.update_session_state("session_1", {"last_employee_code": "EMP2212"})
        
        # We mock LLM to return a specific rewritten query when "his manager" is present
        res = agent.rewrite_query("session_1", "What is his manager's email?")
        
        assert res["needs_clarification"] is False
        assert "EMP1001" in res["standalone_query"]

    def test_clarification_flag(self, agent):
        agent.llm.generate_json_structured = MagicMock(return_value={
            "standalone_query": "Which one?",
            "needs_clarification": True,
            "clarification_question": "Did you mean EMP2212 or EMP1001?"
        })
        
        res = agent.rewrite_query("session_2", "What is his leave?")
        assert res["needs_clarification"] is True
        assert "Did you mean" in res["clarification_question"]

    def test_history_summarization(self, agent):
        # Add 12 turns of history
        for i in range(12):
            agent.db.history.append({"role": "user" if i%2==0 else "assistant", "content": f"Message {i}"})
        
        # Spy on generate_completion to ensure summarization is called
        agent.llm.generate_completion = MagicMock(return_value="Summarized older context.")
        agent.llm.generate_json_structured = MagicMock(return_value={"standalone_query": "test"})
        
        agent.rewrite_query("session_3", "Tell me more")
        
        # The summary function should have been called
        agent.llm.generate_completion.assert_called_once()
