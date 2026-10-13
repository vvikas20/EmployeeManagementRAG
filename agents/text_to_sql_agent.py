from typing import Dict, Any, List
import sqlglot
from sqlglot import parse_one, exp

from database import DatabaseManager
from litellm_client import LiteLLMClient

class TextToSQLAgent:
    """
    Phase 4: Generates SQLite queries from natural language, validates them against
    destructive operations using sqlglot, and limits results to 50 rows.
    """
    def __init__(self, llm: LiteLLMClient, db: DatabaseManager):
        self.llm = llm
        self.db = db

    def generate_and_execute(self, user_query: str) -> Dict[str, Any]:
        """
        1. Fetch safe schema.
        2. Prompt LLM to write a SQLite query.
        3. Validate and sanitize query via sqlglot.
        4. Execute.
        5. Return results.
        """
        schema = self.db.get_safe_schema()
        
        system_prompt = (
            "You are an expert SQLite developer. "
            "Write a valid SQLite query to answer the user's question based on the provided schema.\n\n"
            f"{schema}\n\n"
            "Rules:\n"
            "1. ONLY return the raw SQL query. Do NOT include markdown formatting like ```sql or explanations.\n"
            "2. Do NOT prefix table names with any schema (do not use dbo.).\n"
            "3. Do NOT do complex arithmetic (like calculating percentages or ratios manually) if it can be avoided. Just return raw counts/totals if requested.\n"
            "4. Only query columns that exist in the schema."
        )

        raw_sql = self.llm.generate_completion(
            prompt=user_query,
            system_prompt=system_prompt,
            use_fast_model=False,
            temperature=0.0
        ).strip()

        # Remove markdown if the LLM hallucinated it
        if raw_sql.startswith("```"):
            lines = raw_sql.split("\n")
            if lines[0].startswith("```"): lines = lines[1:]
            if lines[-1].startswith("```"): lines = lines[:-1]
            raw_sql = "\n".join(lines).strip()

        # Validate with sqlglot
        try:
            parsed = parse_one(raw_sql, read="sqlite")
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to parse generated SQL. Invalid syntax. {e}",
                "sql": raw_sql
            }

        # Ensure it's a SELECT statement
        if not isinstance(parsed, exp.Select):
            return {
                "success": False,
                "error": "Only SELECT queries are permitted for safety reasons.",
                "sql": raw_sql
            }

        # Enforce 50 row limit
        if not parsed.args.get("limit"):
            parsed = parsed.limit(50, dialect="sqlite")
            
        safe_sql = parsed.sql(dialect="sqlite")

        try:
            results = self.db.execute_read_only_query(safe_sql)
            return {
                "success": True,
                "sql": safe_sql,
                "results": results
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "sql": safe_sql
            }
