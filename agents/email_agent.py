from litellm_client import LiteLLMClient
from typing import Dict, Any, Optional

class EmailSynthesizerAgent:
    """
    Email Action Agent that crafts professional corporate emails.
    Includes Subject, To (Employee Email), CC (Manager Email), and Body.
    """
    def __init__(self, llm_client: LiteLLMClient):
        self.llm = llm_client

    def generate_email(
        self, 
        user_prompt: str, 
        emp_data: Optional[Dict[str, Any]] = None,
        policy_context: Optional[str] = None
    ) -> Dict[str, Any]:
        
        emp_name = f"{emp_data.get('FirstName', 'Employee')} {emp_data.get('LastName', '')}".strip() if emp_data else "Employee"
        emp_email = emp_data.get('Email', 'employee@corpdemo.local') if emp_data else "employee@corpdemo.local"
        mgr_name = emp_data.get('ManagerName', 'Line Manager') if emp_data else "Line Manager"
        mgr_email = emp_data.get('ManagerEmail', 'manager@corpdemo.local') if emp_data else "manager@corpdemo.local"
        designation = emp_data.get('DesignationName', 'Team Member') if emp_data else "Team Member"
        dept = emp_data.get('DepartmentName', 'Department') if emp_data else "Department"

        system_prompt = (
            "You are an Executive Corporate Communications Assistant.\n"
            "Your task is to draft a professional corporate email EXACTLY matching the user's intent.\n"
            "The email topic, purpose, and tone must directly reflect the user's instruction.\n"
            "DO NOT default to training notifications unless explicitly requested.\n"
            "Return the email ONLY as a single-line valid JSON object with these exact keys:\n"
            '{"subject": "...", "to": "...", "cc": "...", "body": "..."}\n'
            "RULES:\n"
            "- All string values must be on one line (escape newlines as \\n inside the body value).\n"
            "- Do NOT use markdown code fences or backticks.\n"
            "- Do NOT add any text before or after the JSON object.\n"
            "- The 'body' must reflect the SPECIFIC user instruction, not a generic template."
        )

        prompt = (
            f"User Instruction: {user_prompt}\n\n"
            f"Target Employee Details:\n"
            f"- Name: {emp_name}\n"
            f"- Email: {emp_email}\n"
            f"- Designation: {designation} ({dept})\n"
            f"- Manager: {mgr_name} ({mgr_email})\n\n"
            f"Policy Context / References (if applicable):\n{policy_context if policy_context else 'N/A'}\n\n"
            f"Write the email body based on the instruction above. "
            f"The email should be addressed to {emp_name} and its content should be about: {user_prompt}"
        )

        email_json = self.llm.generate_json_structured(prompt, system_prompt, use_fast_model=False)

        # Check if JSON parsing failed entirely
        if email_json.get("error") == "Failed to parse JSON":
            raw_output = email_json.get("raw_output", "")
            print(f"[EmailAgent] JSON parse failed. Raw output: {raw_output[:300]}")
            # Attempt plain-text fallback: ask LLM to just write the body in plain text
            fallback_body = self._generate_plain_text_body(user_prompt, emp_name, mgr_name, designation, dept)
            return {
                "subject": f"RE: {user_prompt[:60]}",
                "to": f"{emp_name} <{emp_email}>",
                "cc": f"{mgr_name} <{mgr_email}>",
                "body": fallback_body
            }

        # Normalize fields in case LLM returned lists or alternative key names
        to_val = email_json.get("to") or f"{emp_name} <{emp_email}>"
        if isinstance(to_val, list):
            to_val = ", ".join(to_val)

        cc_val = email_json.get("cc") or f"{mgr_name} <{mgr_email}>"
        if isinstance(cc_val, list):
            cc_val = ", ".join(cc_val)

        subject_val = email_json.get("subject") or f"RE: {user_prompt[:60]}"

        body_val = (
            email_json.get("body") or 
            email_json.get("email_body") or 
            email_json.get("message") or 
            email_json.get("content")
        )
        if not body_val:
            # Last resort: generate plain text body
            body_val = self._generate_plain_text_body(user_prompt, emp_name, mgr_name, designation, dept)
        if isinstance(body_val, list):
            body_val = "\n".join(body_val)

        return {
            "subject": str(subject_val),
            "to": str(to_val),
            "cc": str(cc_val),
            "body": str(body_val)
        }

    def _generate_plain_text_body(
        self, 
        user_prompt: str, 
        emp_name: str,
        mgr_name: str,
        designation: str,
        dept: str
    ) -> str:
        """Fallback: generate email body as plain text when JSON mode fails."""
        system = (
            "You are a corporate email writer. Write ONLY the email body text "
            "(no subject line, no JSON, no formatting markers). "
            "The email must directly address the specific topic in the user instruction."
        )
        prompt = (
            f"Write a formal corporate email body for the following instruction:\n"
            f"'{user_prompt}'\n\n"
            f"The email is addressed to {emp_name} ({designation}, {dept} Department).\n"
            f"Sign off as 'HR Department'.\n"
            f"DO NOT write about training unless the instruction mentions training."
        )
        return self.llm.generate_completion(prompt, system, use_fast_model=False, temperature=0.3)
