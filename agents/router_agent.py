from litellm_client import LiteLLMClient
from typing import Dict, Any, List
import re

class RouterSupervisorAgent:
    """
    Supervisor / Router Agent that classifies user intent into execution categories:
    - COMPANY_STATS_QUERY: Aggregate queries about total employees, department counts, headcount, company metrics.
    - POLICY_QUERY: Questions only about company policies, leave rules, entitlements.
    - EMPLOYEE_SQL_QUERY: Queries asking for a specific employee's details, leave balances, manager info.
    - HYBRID_POLICY_EMPLOYEE: Queries combining company policy rules with a specific employee (e.g. "As per leave policy how many leaves employee 2212 can apply for?").
    - EMAIL_ACTION: Requests to draft/write an email for an employee.
    """
    def __init__(self, llm_client: LiteLLMClient):
        self.llm = llm_client

    def route_query(self, query: str) -> Dict[str, Any]:
        system_prompt = (
            "You are an Intent Classifier and Supervisor Agent for an Enterprise Employee RAG system.\n"
            "Categorize the user prompt into exactly ONE intent:\n"
            "1. COMPANY_STATS_QUERY: User asks aggregate questions about overall company metrics, e.g. total number of employees, how many employees we have, total departments, headcount.\n"
            "2. EMAIL_ACTION: User wants an email written, drafted, or sent to an employee or manager.\n"
            "3. HYBRID_POLICY_EMPLOYEE: User asks a question about a specific employee's eligible leaves, benefits, or policy application.\n"
            "4. EMPLOYEE_SQL_QUERY: User asks about a specific employee's profile, salary, manager, or DB details.\n"
            "5. CUSTOM_SQL_QUERY: User asks an ad-hoc or complex database question about departments, leaves, etc that doesn't fit the others.\n"
            "6. POLICY_QUERY: General questions about company policy, benefits, working hours.\n\n"
            "Also extract any specific employee codes or numbers mentioned (e.g., '2212', 'EMP2212', '1001', 'EMP0001').\n\n"
            "Respond in JSON format:\n"
            "{\n"
            '  "intent": "COMPANY_STATS_QUERY" | "EMAIL_ACTION" | "HYBRID_POLICY_EMPLOYEE" | "EMPLOYEE_SQL_QUERY" | "CUSTOM_SQL_QUERY" | "POLICY_QUERY",\n'
            '  "extracted_employee_code": "2212" or null,\n'
            '  "reasoning": "brief explanation"\n'
            "}"
        )
        
        result = self.llm.generate_json_structured(
            prompt=f"User Query: {query}",
            system_prompt=system_prompt,
            use_fast_model=True
        )

        intent = result.get("intent")
        emp_code = result.get("extracted_employee_code")

        # Heuristic regex extraction for employee numbers like 2212 or EMP2212
        emp_match = re.search(r'\b(?:EMP)?(\d{3,5})\b', query, re.IGNORECASE)
        if emp_match and not emp_code:
            emp_code = emp_match.group(1)

        query_lower = query.lower()
        if not intent or intent not in ["COMPANY_STATS_QUERY", "EMAIL_ACTION", "HYBRID_POLICY_EMPLOYEE", "EMPLOYEE_SQL_QUERY", "POLICY_QUERY"]:
            if "how many employee" in query_lower or "total employee" in query_lower or "headcount" in query_lower or "how many people" in query_lower or "how many workers" in query_lower:
                intent = "COMPANY_STATS_QUERY"
            elif "email" in query_lower or "write an email" in query_lower or "send an email" in query_lower:
                intent = "EMAIL_ACTION"
            elif emp_code and ("leave" in query_lower or "policy" in query_lower or "apply" in query_lower):
                intent = "HYBRID_POLICY_EMPLOYEE"
            elif emp_code:
                intent = "EMPLOYEE_SQL_QUERY"
            else:
                intent = "POLICY_QUERY"

        return {
            "intent": intent,
            "extracted_employee_code": emp_code,
            "reasoning": result.get("reasoning") or f"Routed to '{intent}' based on keyword/entity matching. Employee code: {emp_code or 'None'}"
        }
