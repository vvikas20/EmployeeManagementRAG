"""Local MCP tools for the Employee Management RAG system."""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from fastapi.encoders import jsonable_encoder
from mcp.server import MCPServer

from agents.sql_tools import SQLToolExecutor
from database import DatabaseManager
from vector_db import VectorDBManager


mcp = MCPServer("employee-management-rag")


@lru_cache(maxsize=1)
def _database() -> DatabaseManager:
    return DatabaseManager()


@lru_cache(maxsize=1)
def _sql_tools() -> SQLToolExecutor:
    return SQLToolExecutor(_database())


@lru_cache(maxsize=1)
def _vector_db() -> VectorDBManager:
    return VectorDBManager()


@lru_cache(maxsize=1)
def _email_agent():
    from agents.email_agent import EmailSynthesizerAgent
    from litellm_client import LiteLLMClient

    return EmailSynthesizerAgent(LiteLLMClient())


def _dispatch_sql_tool(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    result = _sql_tools().dispatch(tool_name, json.dumps(arguments))
    return jsonable_encoder(result)


@mcp.tool()
def get_employee_profile(employee_code: str) -> dict[str, Any]:
    """Get an employee profile by employee number or name."""
    return _dispatch_sql_tool("get_employee_profile", {"employee_code": employee_code})


@mcp.tool()
def get_leave_balance(employee_code: str, leave_type: str | None = None) -> dict[str, Any]:
    """Get an employee's leave balances, optionally filtered to CL, EL, or SL."""
    arguments: dict[str, Any] = {"employee_code": employee_code}
    if leave_type is not None:
        arguments["leave_type"] = leave_type
    return _dispatch_sql_tool("get_leave_balance", arguments)


@mcp.tool()
def get_team_members(manager_code: str) -> dict[str, Any]:
    """Get direct reports for a manager by employee number."""
    return _dispatch_sql_tool("get_team_members", {"manager_code": manager_code})


@mcp.tool()
def get_department_stats(department: str | None = None) -> dict[str, Any]:
    """Get employee headcount by department, optionally filtered by name."""
    arguments = {"department": department} if department is not None else {}
    return _dispatch_sql_tool("get_department_stats", arguments)


@mcp.tool()
def search_policy(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    """Find relevant company policy passages for a question."""
    if not query.strip():
        raise ValueError("query must not be empty")
    if not 1 <= top_k <= 10:
        raise ValueError("top_k must be between 1 and 10")
    return jsonable_encoder(_vector_db().search_policy(query, top_k=top_k))


@mcp.tool()
def draft_employee_email(
    employee_code: str,
    instruction: str,
    include_policy_context: bool = False,
) -> dict[str, Any]:
    """Draft an email to an employee; this tool does not send it."""
    if not instruction.strip():
        raise ValueError("instruction must not be empty")

    employee = _database().get_employee(employee_code)
    if employee is None:
        return {"error": f"employee_not_found: no record for '{employee_code}'"}

    policy_context = ""
    if include_policy_context:
        chunks = _vector_db().search_policy(instruction, top_k=3)
        policy_context = "\n\n".join(chunk["text"] for chunk in chunks)

    draft = _email_agent().generate_email(instruction, employee, policy_context)
    return jsonable_encoder(draft)


if __name__ == "__main__":
    mcp.run(transport="stdio")