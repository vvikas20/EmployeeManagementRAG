from database import MSSQLDatabase
from typing import Dict, Any, Optional

class EmployeeSQLAgent:
    """
    SQL Agent that interacts with local MSSQL EmployeeManagement Database.
    Fetches employee details, manager email, designation, and current leave balances.
    """
    def __init__(self, db: MSSQLDatabase):
        self.db = db

    def get_employee_data(self, emp_identifier: str) -> Optional[Dict[str, Any]]:
        return self.db.get_employee(emp_identifier)

    def format_employee_summary(self, emp_data: Dict[str, Any]) -> str:
        if not emp_data:
            return "No employee details found in the database."

        emp_code = emp_data.get('EmployeeNumber', 'N/A')
        full_name = f"{emp_data.get('FirstName', '')} {emp_data.get('LastName', '')}".strip()
        email = emp_data.get('Email', 'N/A')
        dept = emp_data.get('DepartmentName', 'N/A')
        designation = emp_data.get('DesignationName', 'N/A')
        manager_name = emp_data.get('ManagerName', 'N/A')
        manager_email = emp_data.get('ManagerEmail', 'N/A')

        balances = emp_data.get('LeaveBalances', {})
        cl = balances.get('CL', {}).get('balance', 0) if isinstance(balances.get('CL'), dict) else balances.get('CL', 0)
        el = balances.get('EL', {}).get('balance', 0) if isinstance(balances.get('EL'), dict) else balances.get('EL', 0)
        sl = balances.get('SL', {}).get('balance', 0) if isinstance(balances.get('SL'), dict) else balances.get('SL', 0)
        total_avail = balances.get('TotalAvailable', cl + el + sl)

        return (
            f"Employee: {full_name} (Code: {emp_code})\n"
            f"Designation: {designation} | Department: {dept}\n"
            f"Email: {email}\n"
            f"Manager: {manager_name} ({manager_email})\n"
            f"Current Leave Balances: CL={cl} days, EL={el} days, SL={sl} days (Total Available: {total_avail} days)"
        )

    def get_company_overview(self) -> str:
        """Fetch and format overall company metrics from MSSQL DB."""
        stats = self.db.get_company_stats()
        active = stats.get("active_employees", 0)
        depts = stats.get("department_breakdown", {})
        
        return (
            f"We currently have **{active} active employees** in the company across {len(depts)} departments.\n\n"
            f"**Department Breakdown:**\n"
            + "\n".join([f"- **{name}**: {count} employees" for name, count in depts.items()])
        )
