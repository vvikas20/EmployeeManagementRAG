"""
agents/tool_agent.py
Phase 1 — 4-iteration tool-calling agent loop.

Flow per iteration:
  1. Call default-llm (llama3.1) with current messages + TOOL_SCHEMAS.
  2. If response has tool_calls  → dispatch each via SQLToolExecutor, append tool messages, loop.
  3. If response has plain content → return as final answer immediately.
  4. Hard stop after MAX_ITERATIONS; return last available content or a safe fallback.

Error handling:
  - LLM error     → return {"error": ...} dict immediately with trace step.
  - Invalid JSON args → SQLToolExecutor.dispatch() returns {"error": "invalid_arguments: ..."}.
  - Unknown tool  → SQLToolExecutor.dispatch() returns {"error": "unknown_tool: ..."}.
  - Tool exception → SQLToolExecutor.dispatch() returns {"error": "tool_execution_error: ..."}.
  - All errors are fed back to the model as tool result messages (the model may self-correct).
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List

from litellm_client import LiteLLMClient
from agents.sql_tools import TOOL_SCHEMAS, SQLToolExecutor

MAX_ITERATIONS = 4

_SYSTEM_PROMPT = (
    "You are an expert Enterprise HR AI Assistant with access to live employee database tools.\n"
    "Rules:\n"
    "1. NEVER invent employee data — always call the appropriate tool to retrieve it.\n"
    "2. NEVER perform arithmetic — use the pre-computed totals (e.g. total_available) returned by tools.\n"
    "3. Call only the tools listed. If a tool returns an error, report it honestly.\n"
    "4. When you have all the data needed, write a clear, concise answer."
)


class ToolCallingAgent:
    """
    Drives a multi-turn tool-calling conversation with the default LLM.
    Optionally injects a policy_context string into the system prompt for hybrid queries.
    """

    def __init__(self, llm: LiteLLMClient, tool_executor: SQLToolExecutor):
        self.llm = llm
        self.executor = tool_executor

    def run(self, user_query: str, policy_context: str = "") -> Dict[str, Any]:
        """
        Execute the tool-calling loop for a user query.

        Returns:
            {
                "answer":          str   — final synthesized answer,
                "tool_calls_made": list  — record of every tool dispatched,
                "iterations":      int   — how many LLM turns were used,
                "trace_steps":     list  — trace entries for the orchestrator,
            }
        """
        trace_steps: List[Dict[str, Any]] = []
        tool_calls_made: List[Dict[str, Any]] = []

        # Build system prompt — append policy context for hybrid intents
        system_content = _SYSTEM_PROMPT
        if policy_context:
            system_content += f"\n\nRelevant company policy context (use as background knowledge):\n{policy_context}"

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_content},
            {"role": "user",   "content": user_query},
        ]

        for iteration in range(1, MAX_ITERATIONS + 1):
            step_start = time.time()

            # ── Call LLM with tools ───────────────────────────────────────────
            response_msg = self.llm.generate_with_tools(
                messages=messages,
                tools=TOOL_SCHEMAS,
            )
            step_elapsed = round(time.time() - step_start, 2)

            # generate_with_tools returns a dict with "error" key on failure
            if isinstance(response_msg, dict) and "error" in response_msg:
                trace_steps.append({
                    "step": f"Tool Loop — Iteration {iteration}",
                    "status": "LLM call failed",
                    "success": False,
                    "duration_s": step_elapsed,
                    "details": response_msg["error"],
                })
                return {
                    "answer": f"LLM error during tool-calling loop: {response_msg['error']}",
                    "tool_calls_made": tool_calls_made,
                    "iterations": iteration,
                    "trace_steps": trace_steps,
                }

            # Append assistant message to conversation history
            messages.append({"role": "assistant", **_message_to_dict(response_msg)})

            has_tool_calls = bool(getattr(response_msg, "tool_calls", None))

            # ── No tool call → model produced a final answer ──────────────────
            if not has_tool_calls:
                answer = (getattr(response_msg, "content", None) or "").strip()
                trace_steps.append({
                    "step": f"Tool Loop — Iteration {iteration} (Final Answer)",
                    "status": (
                        f"Model answered directly after {len(tool_calls_made)} tool call(s). "
                        f"Duration: {step_elapsed}s"
                    ),
                    "success": True,
                    "duration_s": step_elapsed,
                })
                return {
                    "answer": answer,
                    "tool_calls_made": tool_calls_made,
                    "iterations": iteration,
                    "trace_steps": trace_steps,
                }

            # ── Execute each tool call ────────────────────────────────────────
            dispatched_this_turn: List[str] = []

            for tc in response_msg.tool_calls:
                tool_name = tc.function.name
                raw_args  = tc.function.arguments  # string from the LLM

                call_start = time.time()
                result     = self.executor.dispatch(tool_name, raw_args)
                call_elapsed = round(time.time() - call_start, 2)

                is_error = "error" in result
                tool_calls_made.append({
                    "tool":      tool_name,
                    "args":      raw_args,
                    "success":   not is_error,
                    "duration_s": call_elapsed,
                    "result_preview": str(result)[:120],
                })
                dispatched_this_turn.append(
                    f"{tool_name}: {'ERROR — ' + result['error'] if is_error else 'OK'}"
                )

                # Feed the tool result back as a tool message
                messages.append({
                    "role":         "tool",
                    "tool_call_id": tc.id,
                    "name":         tool_name,
                    "content":      json.dumps(result, default=str),
                })

            trace_steps.append({
                "step": f"Tool Loop — Iteration {iteration}",
                "status": f"Executed: {', '.join(dispatched_this_turn)} | Duration: {step_elapsed}s",
                "success": True,
                "duration_s": step_elapsed,
                "details": tool_calls_made[-(len(response_msg.tool_calls)):],
            })

        # ── Iteration cap hit ─────────────────────────────────────────────────
        # Return the last assistant content we received (may be partial / empty)
        last_content = ""
        for m in reversed(messages):
            if m.get("role") == "assistant" and m.get("content"):
                last_content = m["content"]
                break

        trace_steps.append({
            "step": "Tool Loop — Iteration Cap Reached",
            "status": f"Stopped after {MAX_ITERATIONS} iterations without a final answer.",
            "success": False,
        })

        return {
            "answer": last_content.strip() or (
                "I was unable to produce a complete answer within the allowed number of steps. "
                "Please try rephrasing your query."
            ),
            "tool_calls_made": tool_calls_made,
            "iterations": MAX_ITERATIONS,
            "trace_steps": trace_steps,
        }

    def run_stream(self, user_query: str, policy_context: str = ""):
        """
        Streaming version of run(). Yields trace steps and final answer tokens.
        """
        trace_steps: List[Dict[str, Any]] = []
        tool_calls_made: List[Dict[str, Any]] = []

        system_content = _SYSTEM_PROMPT
        if policy_context:
            system_content += f"\n\nRelevant company policy context (use as background knowledge):\n{policy_context}"

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_content},
            {"role": "user",   "content": user_query},
        ]

        for iteration in range(1, MAX_ITERATIONS + 1):
            step_start = time.time()

            yield {"type": "step", "step": f"Tool Loop — Iteration {iteration}", "status": "Calling LLM..."}

            # For streaming tool calls, litellm returns a generator
            response_generator = self.llm.generate_with_tools_stream(
                messages=messages,
                tools=TOOL_SCHEMAS,
            )

            if isinstance(response_generator, dict) and "error" in response_generator:
                err_msg = response_generator["error"]
                trace_steps.append({
                    "step": f"Tool Loop — Iteration {iteration}",
                    "status": "LLM call failed",
                    "success": False,
                    "duration_s": round(time.time() - step_start, 2),
                    "details": err_msg,
                })
                yield {"type": "step", "step": f"Tool Loop — Iteration {iteration}", "status": "LLM call failed"}
                yield {"type": "complete", "result": {
                    "answer": f"LLM error: {err_msg}",
                    "tool_calls_made": tool_calls_made,
                    "iterations": iteration,
                    "trace_steps": trace_steps
                }}
                return

            # Process the stream
            current_tool_calls = {}
            final_content = ""
            
            for chunk in response_generator:
                delta = chunk.choices[0].delta
                if getattr(delta, "tool_calls", None):
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in current_tool_calls:
                            current_tool_calls[idx] = {"id": tc.id, "function": {"name": tc.function.name, "arguments": ""}}
                        if tc.function.arguments:
                            current_tool_calls[idx]["function"]["arguments"] += tc.function.arguments
                
                content = getattr(delta, "content", None)
                if content:
                    final_content += content
                    yield {"type": "token", "content": content}
            
            step_elapsed = round(time.time() - step_start, 2)

            # Build the assistant message
            assistant_msg = {"role": "assistant", "content": final_content if final_content else None}
            tool_calls_list = list(current_tool_calls.values())
            if tool_calls_list:
                assistant_msg["tool_calls"] = []
                for tc in tool_calls_list:
                    assistant_msg["tool_calls"].append({
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"]
                        }
                    })
            
            messages.append(assistant_msg)

            # No tool calls means we got a final answer
            if not tool_calls_list:
                trace_steps.append({
                    "step": f"Tool Loop — Iteration {iteration} (Final Answer)",
                    "status": f"Model answered directly after {len(tool_calls_made)} tool call(s). Duration: {step_elapsed}s",
                    "success": True,
                    "duration_s": step_elapsed,
                })
                yield {"type": "complete", "result": {
                    "answer": final_content.strip(),
                    "tool_calls_made": tool_calls_made,
                    "iterations": iteration,
                    "trace_steps": trace_steps,
                }}
                return

            # Execute tool calls
            dispatched_this_turn: List[str] = []
            for tc in tool_calls_list:
                tool_name = tc["function"]["name"]
                raw_args = tc["function"]["arguments"]
                
                yield {"type": "step", "step": f"Tool Loop — Iteration {iteration}", "status": f"Executing tool: {tool_name}"}

                call_start = time.time()
                result = self.executor.dispatch(tool_name, raw_args)
                call_elapsed = round(time.time() - call_start, 2)

                is_error = "error" in result
                tool_calls_made.append({
                    "tool": tool_name,
                    "args": raw_args,
                    "success": not is_error,
                    "duration_s": call_elapsed,
                    "result_preview": str(result)[:120],
                })
                dispatched_this_turn.append(f"{tool_name}: {'ERROR' if is_error else 'OK'}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": tool_name,
                    "content": json.dumps(result, default=str),
                })
            
            trace_steps.append({
                "step": f"Tool Loop — Iteration {iteration}",
                "status": f"Executed: {', '.join(dispatched_this_turn)} | Duration: {step_elapsed}s",
                "success": True,
                "duration_s": step_elapsed,
                "details": tool_calls_made[-(len(tool_calls_list)):],
            })

        # Cap hit
        trace_steps.append({
            "step": "Tool Loop — Iteration Cap Reached",
            "status": f"Stopped after {MAX_ITERATIONS} iterations without a final answer.",
            "success": False,
        })
        
        last_content = ""
        for m in reversed(messages):
            if m.get("role") == "assistant" and m.get("content"):
                last_content = m["content"]
                break

        yield {"type": "complete", "result": {
            "answer": last_content.strip() or "Stopped before completing.",
            "tool_calls_made": tool_calls_made,
            "iterations": MAX_ITERATIONS,
            "trace_steps": trace_steps,
        }}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _message_to_dict(msg: Any) -> Dict[str, Any]:
    """
    Convert a LiteLLM Message object into a plain dict suitable for appending
    to the messages list. Preserves tool_calls if present.
    """
    d: Dict[str, Any] = {"content": getattr(msg, "content", None)}
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        d["tool_calls"] = [
            {
                "id":   tc.id,
                "type": "function",
                "function": {
                    "name":      tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in tool_calls
        ]
    return d
