"""
tests/test_phase3.py
Phase 3 — pytest suite for streaming functionality.

Mocks the LLM and DB to verify SSE yielding works in orchestrator.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock
import pytest

from agents.orchestrator import MultiAgentOrchestrator

class StubLLM:
    def __init__(self):
        pass

    def generate_completion_stream(self, prompt, system_prompt="", use_fast_model=False):
        yield "Streamed "
        yield "token "
        yield "123."

    def generate_json_structured(self, *args, **kwargs):
        # Stub router
        return {"intent": "POLICY_QUERY", "extracted_employee_code": None, "reasoning": "mock"}

class StubPolicyAgent:
    def retrieve_context(self, *args, **kwargs):
        return [{"text": "Mock policy", "similarity": 0.9}]
    
    def answer_policy_query(self, *args, **kwargs):
        return {"answer": "Static answer (not used in stream)"}

class TestStreaming:
    @pytest.fixture
    def orchestrator(self):
        orch = MultiAgentOrchestrator.__new__(MultiAgentOrchestrator)
        orch.use_tool_calling = False
        orch.use_conversation_context = False
        orch.use_text_to_sql = False
        orch.llm = StubLLM()
        orch.policy_agent = StubPolicyAgent()
        
        # Stub router
        orch.router = MagicMock()
        orch.router.route_query.return_value = {"intent": "POLICY_QUERY", "extracted_employee_code": None, "reasoning": "mock"}
        
        orch.memory_agent = MagicMock()
        return orch

    def test_process_query_stream_yields_tokens(self, orchestrator):
        generator = orchestrator.process_query_stream("What is the leave policy?")
        
        events = list(generator)
        
        # We expect some steps, then some tokens, then a complete event
        step_events = [e for e in events if e["type"] == "step"]
        token_events = [e for e in events if e["type"] == "token"]
        complete_events = [e for e in events if e["type"] == "complete"]
        
        assert len(step_events) > 0
        assert len(token_events) == 3
        assert token_events[0]["content"] == "Streamed "
        assert token_events[1]["content"] == "token "
        assert token_events[2]["content"] == "123."
        
        assert len(complete_events) == 1
        assert complete_events[0]["result"]["answer"] == "Streamed token 123."
