import os
import pytest
from unittest.mock import MagicMock
from database import DatabaseManager
from agents.sql_tools import SQLToolExecutor
from agents.text_to_sql_agent import TextToSQLAgent


@pytest.fixture
def test_db(tmp_path):
    db_file = str(tmp_path / "test_employee.db")
    # Patch config path or pass custom path directly
    db = DatabaseManager.__new__(DatabaseManager)
    db.config_path = "config.yaml"
    db.config = {"path": db_file}
    db.db_path = db_file
    db.database = os.path.basename(db_file)
    db._initialize_db()
    return db


def test_schema_initialization_and_seeding(test_db):
    assert test_db.test_connection() is True
    
    # Verify tables exist
    with test_db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        required = {"Departments", "Designations", "Locations", "LeaveTypes", "Employees", "EmployeeLeaveBalances", "ChatSessions", "ChatMessages"}
        assert required.issubset(tables)


def test_get_employee_by_id_and_code(test_db):
    # Fetch EMP2212
    emp = test_db.get_employee("EMP2212")
    assert emp is not None
    assert emp["EmployeeNumber"] == "EMP2212"
    assert emp["FirstName"] == "Vikram"
    assert emp["LastName"] == "Sharma"
    assert emp["DepartmentName"] == "Engineering & Technology"
    assert emp["DesignationName"] == "Senior Software Engineer"
    assert emp["ManagerName"] == "Aarav Agarwal"
    assert "LeaveBalances" in emp
    assert emp["LeaveBalances"]["CL"]["balance"] == 8.0
    assert emp["LeaveBalances"]["TotalAvailable"] == 46.0

    # Fetch by numeric code "2212"
    emp_by_num = test_db.get_employee("2212")
    assert emp_by_num is not None
    assert emp_by_num["EmployeeNumber"] == "EMP2212"

    # Fetch manager 1001
    mgr = test_db.get_employee("EMP1001")
    assert mgr is not None
    assert mgr["FirstName"] == "Aarav"


def test_get_employee_by_name(test_db):
    emp = test_db.get_employee("Vikram")
    assert emp is not None
    assert emp["EmployeeNumber"] == "EMP2212"


def test_company_stats(test_db):
    stats = test_db.get_company_stats()
    assert stats["total_employees"] >= 6
    assert stats["active_employees"] >= 6
    assert stats["total_departments"] >= 3
    assert "Engineering & Technology" in stats["department_breakdown"]


def test_chat_logging_and_history(test_db):
    session_id = "test-session-123"
    test_db.log_chat_message(session_id, "user", "Hello there!")
    test_db.log_chat_message(session_id, "assistant", "Hello! How can I help?")

    history = test_db.get_chat_history(session_id, limit=5)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "Hello there!"
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == "Hello! How can I help?"


def test_safe_schema_excludes_salary(test_db):
    schema = test_db.get_safe_schema()
    assert "Database Schema (SQLite):" in schema
    assert "Table: Employees" in schema
    assert "Salary" not in schema
    assert "DepartmentName" in schema or "Table: Departments" in schema


def test_read_only_query_execution(test_db):
    results = test_db.execute_read_only_query("SELECT EmployeeNumber, FirstName FROM Employees WHERE EmployeeNumber = 'EMP2212'")
    assert len(results) == 1
    assert results[0]["EmployeeNumber"] == "EMP2212"
    assert results[0]["FirstName"] == "Vikram"


def test_sql_tools_executor(test_db):
    executor = SQLToolExecutor(test_db)
    
    # Test employee profile tool
    res = executor.dispatch("get_employee_profile", '{"employee_code": "EMP2212"}')
    assert "error" not in res
    assert res["FirstName"] == "Vikram"

    # Test leave balance tool
    leave_res = executor.dispatch("get_leave_balance", '{"employee_code": "EMP2212", "leave_type": "CL"}')
    assert "error" not in leave_res
    assert leave_res["leave_type"] == "CL"
    assert leave_res["balance_days"] == 8.0

    # Test team members tool
    team_res = executor.dispatch("get_team_members", '{"manager_code": "EMP1001"}')
    assert "error" not in team_res
    assert team_res["team_size"] >= 2

    # Test department stats tool
    dept_res = executor.dispatch("get_department_stats", '{"department": "Engineering"}')
    assert "error" not in dept_res
    assert dept_res["total_employees"] >= 3


def test_text_to_sql_agent(test_db):
    mock_llm = MagicMock()
    mock_llm.generate_completion.return_value = "SELECT EmployeeNumber, FirstName FROM Employees WHERE DepartmentId = 1"
    
    agent = TextToSQLAgent(mock_llm, test_db)
    result = agent.generate_and_execute("Show all engineering employees")
    
    assert result["success"] is True
    assert "LIMIT 50" in result["sql"]
    assert len(result["results"]) >= 3
