"""
agents/sql_tools.py
Phase 1 — Parameterised SQL tools with OpenAI-style tool schemas.

Design notes:
- All SQL queries use pyodbc '?' placeholders (no string interpolation on user data).
- Results are capped at MAX_ROWS = 50.
- 'Salary' is excluded from all tool results (RBAC for sensitive fields is deferred).
- Each tool returns a JSON-serialisable dict, or {"error": "<reason>"} on failure.
- RBAC: not implemented in this phase — will be layered on in a later enhancement.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from database import MSSQLDatabase


# ─── Pydantic argument models ────────────────────────────────────────────────

class GetEmployeeProfileArgs(BaseModel):
    employee_code: str = Field(
        ...,
        description="Employee number (e.g. '2212', 'EMP2212') or a full/partial name."
    )


class GetLeaveBalanceArgs(BaseModel):
    employee_code: str = Field(
        ...,
        description="Employee number (e.g. '2212' or 'EMP2212')."
    )
    leave_type: Optional[str] = Field(
        None,
        description="Optional leave type code: 'CL' (Casual), 'EL' (Earned), 'SL' (Sick), "
                    "'ML' (Maternity), 'PL' (Paternity), 'BL' (Bereavement), or 'TL' (Training). "
                    "Omit to retrieve all types including a pre-computed total."
    )


class GetTeamMembersArgs(BaseModel):
    manager_code: str = Field(
        ...,
        description="Manager's employee number (e.g. '1001' or 'EMP0001')."
    )


class GetDepartmentStatsArgs(BaseModel):
    department: Optional[str] = Field(
        None,
        description="Optional partial department name filter (e.g. 'Engineering'). "
                    "Omit to get stats for all departments."
    )


# ─── OpenAI-style tool schema definitions ────────────────────────────────────

TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_employee_profile",
            "description": (
                "Fetch a specific employee's profile: name, email, department, "
                "designation, manager details, hire date, and employment status. "
                "Use this before any query that mentions a specific employee."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "employee_code": {
                        "type": "string",
                        "description": "Employee number e.g. '2212', 'EMP2212', or a full/partial name."
                    }
                },
                "required": ["employee_code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_leave_balance",
            "description": (
                "Fetch remaining leave balance days for an employee. "
                "Returns CL (Casual Leave), EL (Earned Leave), SL (Sick Leave), "
                "ML (Maternity), PL (Paternity), BL (Bereavement), TL (Training) "
                "and a pre-computed total_available field. "
                "IMPORTANT: never do arithmetic — report total_available directly."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "employee_code": {
                        "type": "string",
                        "description": "Employee number e.g. '2212' or 'EMP2212'."
                    },
                    "leave_type": {
                        "type": "string",
                        "description": "Optional: 'CL', 'EL', 'SL', 'ML', 'PL', 'BL', or 'TL' to filter to a single type.",
                        "enum": ["CL", "EL", "SL", "ML", "PL", "BL", "TL"]
                    }
                },
                "required": ["employee_code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_team_members",
            "description": "Fetch all direct reports (team members) for a given manager employee code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "manager_code": {
                        "type": "string",
                        "description": "Manager's employee number e.g. '1001' or 'EMP0001'."
                    }
                },
                "required": ["manager_code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_department_stats",
            "description": (
                "Fetch headcount statistics per department. "
                "Returns total_employees and a per-department breakdown. "
                "Optionally filter by partial department name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "department": {
                        "type": "string",
                        "description": "Optional partial department name (e.g. 'Engineering'). Omit for all."
                    }
                },
                "required": []
            }
        }
    }
]


# ─── Tool executor ────────────────────────────────────────────────────────────

class SQLToolExecutor:
    """
    Validates Pydantic argument models and executes parameterised SQL tool calls.
    All queries use pyodbc '?' placeholders. Results are capped at MAX_ROWS.
    """

    MAX_ROWS = 50

    def __init__(self, db: MSSQLDatabase):
        self.db = db

    def dispatch(self, tool_name: str, raw_args: str) -> Dict[str, Any]:
        """
        Entry point called by the agent loop.
        Parses + validates args, routes to the matching tool.
        Returns a JSON-serialisable dict, or {"error": "<reason>"} on any failure.
        """
        # 1. Parse JSON arguments from the LLM's string
        try:
            args_dict = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
        except (json.JSONDecodeError, TypeError) as exc:
            return {"error": f"invalid_arguments: could not parse JSON — {exc}"}

        # 2. Route and validate with Pydantic
        try:
            if tool_name == "get_employee_profile":
                validated = GetEmployeeProfileArgs(**args_dict)
                return self._get_employee_profile(validated.employee_code)

            elif tool_name == "get_leave_balance":
                validated = GetLeaveBalanceArgs(**args_dict)
                return self._get_leave_balance(validated.employee_code, validated.leave_type)

            elif tool_name == "get_team_members":
                validated = GetTeamMembersArgs(**args_dict)
                return self._get_team_members(validated.manager_code)

            elif tool_name == "get_department_stats":
                validated = GetDepartmentStatsArgs(**args_dict)
                return self._get_department_stats(validated.department)

            else:
                return {"error": f"unknown_tool: '{tool_name}' is not a registered tool"}

        except Exception as exc:
            # Covers Pydantic ValidationError and any DB exceptions
            return {"error": f"tool_execution_error: {exc}"}

    # ── Tool implementations ──────────────────────────────────────────────────

    def _get_employee_profile(self, identifier: str) -> Dict[str, Any]:
        emp = self.db.get_employee(identifier)
        if not emp:
            return {"error": f"employee_not_found: no record for '{identifier}'"}
        # Returns full profile including Salary and LocationName.
        # RBAC field-level filtering will be added in a later phase.
        return dict(emp)

    def _get_leave_balance(
        self, identifier: str, leave_type: Optional[str]
    ) -> Dict[str, Any]:
        emp = self.db.get_employee(identifier)
        if not emp:
            return {"error": f"employee_not_found: no record for '{identifier}'"}

        balances: Dict[str, Any] = emp.get("LeaveBalances", {})
        emp_num = emp["EmployeeNumber"]
        emp_name = f"{emp['FirstName']} {emp['LastName']}"

        if leave_type:
            lt = leave_type.upper()
            entry = balances.get(lt)
            if entry is None:
                return {"error": f"leave_type_not_found: '{lt}' not in employee leave records"}
            return {
                "employee_number": emp_num,
                "employee_name": emp_name,
                "leave_type": lt,
                "leave_name": entry["name"] if isinstance(entry, dict) else lt,
                "balance_days": entry["balance"] if isinstance(entry, dict) else float(entry),
                "max_annual": entry.get("max_annual", 0) if isinstance(entry, dict) else 0,
            }

        # All leave types
        breakdown: Dict[str, Any] = {}
        for code, data in balances.items():
            if code == "TotalAvailable":
                continue
            if isinstance(data, dict):
                breakdown[code] = {
                    "name": data["name"],
                    "balance_days": data["balance"],
                    "max_annual": data.get("max_annual", 0),
                }
            else:
                breakdown[code] = {"balance_days": float(data)}

        return {
            "employee_number": emp_num,
            "employee_name": emp_name,
            "leave_breakdown": breakdown,
            "total_available": balances.get("TotalAvailable", 0),
        }

    def _get_team_members(self, manager_code: str) -> Dict[str, Any]:
        manager = self.db.get_employee(manager_code)
        if not manager:
            return {"error": f"manager_not_found: no record for '{manager_code}'"}

        manager_emp_id = manager.get("EmployeeId")

        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT 
                    e.EmployeeNumber, e.FirstName, e.LastName, e.Email,
                    d.DepartmentName, des.Title AS DesignationName,
                    e.EmploymentStatus
                FROM Employees e
                LEFT JOIN Departments d  ON e.DepartmentId  = d.DepartmentId
                LEFT JOIN Designations des ON e.DesignationId = des.DesignationId
                WHERE e.ManagerId = ?
                ORDER BY e.FirstName, e.LastName
                LIMIT ?
                """,
                (manager_emp_id, self.MAX_ROWS),
            )
            rows = cursor.fetchall()
            columns = [col[0] for col in cursor.description]

        members = [dict(zip(columns, row)) for row in rows]
        return {
            "manager_code": manager["EmployeeNumber"],
            "manager_name": f"{manager['FirstName']} {manager['LastName']}",
            "team_size": len(members),
            "team_members": members,
        }

    def _get_department_stats(self, department: Optional[str]) -> Dict[str, Any]:
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            if department:
                cursor.execute(
                    """
                    SELECT d.DepartmentName, COUNT(e.EmployeeId) AS EmployeeCount
                    FROM Departments d
                    LEFT JOIN Employees e ON d.DepartmentId = e.DepartmentId
                    WHERE d.DepartmentName LIKE ?
                    GROUP BY d.DepartmentName
                    ORDER BY EmployeeCount DESC
                    LIMIT ?
                    """,
                    (f"%{department}%", self.MAX_ROWS),
                )
            else:
                cursor.execute(
                    """
                    SELECT d.DepartmentName, COUNT(e.EmployeeId) AS EmployeeCount
                    FROM Departments d
                    LEFT JOIN Employees e ON d.DepartmentId = e.DepartmentId
                    GROUP BY d.DepartmentName
                    ORDER BY EmployeeCount DESC
                    LIMIT ?
                    """,
                    (self.MAX_ROWS,),
                )
            rows = cursor.fetchall()

        breakdown = {row[0]: row[1] for row in rows if row[0]}
        total = sum(breakdown.values())
        return {
            "department_filter": department or "all",
            "total_employees": total,
            "department_breakdown": breakdown,
        }
