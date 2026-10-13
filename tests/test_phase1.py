"""
tests/test_phase1.py
Phase 1 — pytest suite for sql_tools.py and tool_agent.py.

All tests mock litellm.completion so no real LLM or database is needed.

Run with:
    cd "c:\\D\\Playground\\Gen AI\\EmployeeManagementRAGSystem"
    .venv\\Scripts\\python.exe -m pytest tests/test_phase1.py -v
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

# ─── Minimal stubs so imports work without a real DB / config ─────────────────

# Stub MSSQLDatabase so we never need a live connection
class StubDB:
    """Minimal stub returning controlled employee data."""

    _EMP = {
        "EmployeeId":      2212,
        "EmployeeNumber":  "EMP2212",
        "FirstName":       "Vikram",
        "LastName":        "Sharma",
        "Email":           "vikram.sharma2212@corpdemo.local",
        "DepartmentName":  "Engineering & Technology",
        "DesignationName": "Senior Software Engineer",
        "LocationName":    "Bengaluru HQ",
        "ManagerId":       1001,
        "ManagerName":     "Aarav Agarwal",
        "ManagerEmail":    "aarav.agarwal1@corpdemo.local",
        "EmploymentStatus": "Active",
        "Salary":          1200000.0,
        "LeaveBalances": {
            "CL":  {"name": "Casual Leave",  "balance": 8.0,  "max_annual": 12.0},
            "EL":  {"name": "Earned Leave",  "balance": 18.0, "max_annual": 18.0},
            "SL":  {"name": "Sick Leave",    "balance": 10.0, "max_annual": 12.0},
            "TotalAvailable": 36.0,
        },
    }

    def get_employee(self, identifier: str):
        if str(identifier) in ("2212", "EMP2212"):
            return dict(self._EMP)
        return None

    def get_connection(self):
        raise NotImplementedError("StubDB.get_connection not wired in this test")


# ─── Helpers to build fake LiteLLM response objects ──────────────────────────

class _FuncCall:
    def __init__(self, name: str, args: str):
        self.name      = name
        self.arguments = args


class _ToolCall:
    def __init__(self, id_: str, name: str, args: str):
        self.id       = id_
        self.function = _FuncCall(name, args)


class _Message:
    def __init__(self, content: Optional[str] = None, tool_calls: Optional[list] = None):
        self.content    = content
        self.tool_calls = tool_calls or []


class _Choice:
    def __init__(self, message: _Message):
        self.message = message


class _Completion:
    def __init__(self, message: _Message):
        self.choices = [_Choice(message)]


def _make_tool_response(name: str, args: dict, tc_id: str = "tc_001") -> _Completion:
    """Build a fake litellm.completion response that contains a single tool call."""
    return _Completion(
        _Message(
            content=None,
            tool_calls=[_ToolCall(tc_id, name, json.dumps(args))],
        )
    )


def _make_text_response(text: str) -> _Completion:
    """Build a fake litellm.completion response with plain text content."""
    return _Completion(_Message(content=text, tool_calls=[]))


# ─── SQLToolExecutor tests ────────────────────────────────────────────────────

class TestSQLToolExecutor:
    """Unit tests for SQLToolExecutor — no LLM involved."""

    @pytest.fixture
    def executor(self):
        from agents.sql_tools import SQLToolExecutor
        return SQLToolExecutor(StubDB())

    # ── 1. Correct tool selection ─────────────────────────────────────────────

    def test_get_employee_profile_happy_path(self, executor):
        result = executor.dispatch("get_employee_profile", json.dumps({"employee_code": "2212"}))
        assert result["EmployeeNumber"] == "EMP2212"
        assert result["FirstName"] == "Vikram"

    def test_get_employee_profile_includes_salary_and_location(self, executor):
        """Salary and LocationName are now included in full profile (RBAC deferred)."""
        result = executor.dispatch("get_employee_profile", json.dumps({"employee_code": "EMP2212"}))
        assert "Salary" in result
        assert result["Salary"] == 1200000.0
        assert "LocationName" in result
        assert result["LocationName"] == "Bengaluru HQ"

    def test_get_leave_balance_all_types(self, executor):
        result = executor.dispatch("get_leave_balance", json.dumps({"employee_code": "2212"}))
        assert result["total_available"] == 36.0
        assert "CL" in result["leave_breakdown"]
        assert "EL" in result["leave_breakdown"]
        assert "SL" in result["leave_breakdown"]

    def test_get_leave_balance_single_type(self, executor):
        result = executor.dispatch(
            "get_leave_balance",
            json.dumps({"employee_code": "2212", "leave_type": "CL"}),
        )
        assert result["leave_type"] == "CL"
        assert result["balance_days"] == 8.0

    # ── 2. Employee not found ─────────────────────────────────────────────────

    def test_employee_not_found(self, executor):
        result = executor.dispatch("get_employee_profile", json.dumps({"employee_code": "9999"}))
        assert "error" in result
        assert "employee_not_found" in result["error"]

    # ── 3. Malformed / invalid arguments ─────────────────────────────────────

    def test_malformed_json_args(self, executor):
        result = executor.dispatch("get_employee_profile", "NOT VALID JSON {{{{")
        assert "error" in result
        assert "invalid_arguments" in result["error"]

    def test_missing_required_arg(self, executor):
        """employee_code is required — Pydantic should raise ValidationError."""
        result = executor.dispatch("get_leave_balance", json.dumps({}))
        assert "error" in result
        assert "tool_execution_error" in result["error"]

    # ── 4. Unknown tool ───────────────────────────────────────────────────────

    def test_unknown_tool_name(self, executor):
        result = executor.dispatch("drop_all_tables", json.dumps({}))
        assert "error" in result
        assert "unknown_tool" in result["error"]


# ─── ToolCallingAgent tests ───────────────────────────────────────────────────

class TestToolCallingAgent:
    """Integration tests for the agent loop — LLM is mocked via patch."""

    @pytest.fixture
    def agent(self):
        """Create a ToolCallingAgent with a real (but offline) LiteLLMClient stub."""
        # We patch litellm.completion inside litellm_client so the client never
        # actually calls the network.
        from agents.sql_tools import SQLToolExecutor
        from agents.tool_agent import ToolCallingAgent

        # Minimal LiteLLMClient that reads no real config
        with patch("builtins.open", create=True), \
             patch("yaml.safe_load", return_value={
                 "model_list": [
                     {"model_name": "default-llm", "litellm_params": {"model": "ollama/llama3.1", "api_base": "http://localhost:11434"}},
                     {"model_name": "fast-llm",    "litellm_params": {"model": "ollama/qwen2.5:1.5b", "api_base": "http://localhost:11434"}},
                 ]
             }):
            from litellm_client import LiteLLMClient
            llm = LiteLLMClient.__new__(LiteLLMClient)
            llm.default_model = "ollama/llama3.1"
            llm.fast_model    = "ollama/qwen2.5:1.5b"
            llm.api_base      = "http://localhost:11434"
            import litellm as _litellm
            _litellm.telemetry  = False
            _litellm.drop_params = True

        executor = SQLToolExecutor(StubDB())
        return ToolCallingAgent(llm, executor)

    # ── 5. Correct tool selection — model calls tool then answers ─────────────

    @patch("litellm.completion")
    def test_model_calls_tool_then_answers(self, mock_completion, agent):
        """
        Turn 1: model emits get_employee_profile tool call.
        Turn 2: model answers with plain text.
        """
        mock_completion.side_effect = [
            _make_tool_response("get_employee_profile", {"employee_code": "2212"}),
            _make_text_response("Vikram Sharma is a Senior Software Engineer."),
        ]
        result = agent.run("Tell me about employee 2212")
        assert "Vikram" in result["answer"]
        assert result["iterations"] == 2
        assert result["tool_calls_made"][0]["tool"] == "get_employee_profile"
        assert result["tool_calls_made"][0]["success"] is True

    # ── 6. Direct answer without any tool call ────────────────────────────────

    @patch("litellm.completion")
    def test_direct_answer_no_tool_call(self, mock_completion, agent):
        """Model returns content immediately without calling any tool."""
        mock_completion.return_value = _make_text_response(
            "The leave policy allows 36 days total."
        )
        result = agent.run("What is the total leave?")
        assert result["iterations"] == 1
        assert len(result["tool_calls_made"]) == 0
        assert "36" in result["answer"]

    # ── 7. Malformed tool arguments from LLM ─────────────────────────────────

    @patch("litellm.completion")
    def test_malformed_tool_arguments_fed_back(self, mock_completion, agent):
        """
        Model emits a tool call with invalid JSON args.
        The error is fed back as a tool message; model then answers directly.
        """
        bad_tc = _ToolCall("tc_bad", "get_employee_profile", "GARBAGE{{{")
        turn1  = _Completion(_Message(content=None, tool_calls=[bad_tc]))
        turn2  = _make_text_response("I encountered an error retrieving the employee.")

        mock_completion.side_effect = [turn1, turn2]
        result = agent.run("Show employee details for 2212")

        assert result["tool_calls_made"][0]["success"] is False
        assert result["answer"] != ""   # Model still produces an answer

    # ── 8. Iteration cap ──────────────────────────────────────────────────────

    @patch("litellm.completion")
    def test_iteration_cap(self, mock_completion, agent):
        """Model keeps emitting tool calls forever — must stop at MAX_ITERATIONS=4."""
        # Every call returns a tool call (model never settles)
        mock_completion.return_value = _make_tool_response(
            "get_employee_profile", {"employee_code": "2212"}
        )
        result = agent.run("Loop forever")
        assert result["iterations"] == 4
        # Cap trace step must be present
        cap_steps = [s for s in result["trace_steps"] if "Cap" in s["step"]]
        assert len(cap_steps) == 1
        assert cap_steps[0]["success"] is False

    # ── 9. Unknown tool name emitted by model ─────────────────────────────────

    @patch("litellm.completion")
    def test_unknown_tool_error_fed_back(self, mock_completion, agent):
        """
        Model asks for a tool that doesn't exist.
        The error result is fed back; model then answers directly.
        """
        unknown_tc = _ToolCall("tc_x", "delete_employee", json.dumps({"employee_code": "2212"}))
        turn1 = _Completion(_Message(content=None, tool_calls=[unknown_tc]))
        turn2 = _make_text_response("I cannot delete employees with the available tools.")

        mock_completion.side_effect = [turn1, turn2]
        result = agent.run("Delete employee 2212")

        assert result["tool_calls_made"][0]["success"] is False
        assert "unknown_tool" in result["tool_calls_made"][0]["result_preview"]
        assert result["answer"] != ""
