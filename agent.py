#!/usr/bin/env python3
"""Minimal reference agent for E-Commerce Last Exam benchmark.

This file demonstrates the agent/environment interface. In real evaluations it can be replaced by:
- mini-swe-agent (general-purpose CLI agent)
- a custom LLM agent (calling a model via an OpenAI-compatible API)

Agent responsibilities:
1. Read /app/tool_defs.json for the available tool definitions (OpenAI function-calling schema)
2. Read system.md + instruction.md for the task prompt
3. Call the LLM to decide -> call tools -> loop until the task is done
4. Write the final result to /app/answer.json

Tool call convention:
  /app/tools/<tool_name> --param1 value1 --param2 value2
  Output is JSON (stdout)
"""
import json
import os
import subprocess
import sys

# Environment file paths (inside the container)
TOOL_DEFS_PATH = "/app/tool_defs.json"
SYSTEM_MD_PATH = "/app/system.md"
INSTRUCTION_MD_PATH = "/app/instruction.md"
ANSWER_PATH = "/app/answer.json"
TOOL_CALL_LOG = os.environ.get("TOOL_CALL_LOG", "/app/.tool_calls.jsonl")

# Agent LLM config (injected via environment variables)
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:4000")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3.7-plus")


def load_tool_defs():
    with open(TOOL_DEFS_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_prompts():
    system = ""
    instruction = ""
    if os.path.exists(SYSTEM_MD_PATH):
        with open(SYSTEM_MD_PATH, encoding="utf-8") as f:
            system = f.read()
    if os.path.exists(INSTRUCTION_MD_PATH):
        with open(INSTRUCTION_MD_PATH, encoding="utf-8") as f:
            instruction = f.read()
    return system, instruction


def call_tool(tool_name, arguments):
    """Execute a single tool call and return the output JSON."""
    tool_path = f"/app/tools/{tool_name}"
    if not os.path.exists(tool_path):
        return {"error": f"tool not found: {tool_name}"}

    cmd = [tool_path]
    for k, v in arguments.items():
        cmd += [f"--{k}", str(v)]

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            return {"error": r.stderr[:500]}
        try:
            return json.loads(r.stdout)
        except json.JSONDecodeError:
            return {"raw_output": r.stdout[:2000]}
    except subprocess.TimeoutExpired:
        return {"error": "tool call timeout"}


def log_tool_call(tool_name, arguments, result):
    """Log a tool call."""
    entry = {"tool": tool_name, "arguments": arguments, "result": result}
    with open(TOOL_CALL_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def call_llm(messages, tools):
    """Call the LLM API (OpenAI-compatible)."""
    try:
        import openai
    except ImportError:
        print("ERROR: openai package not installed", file=sys.stderr)
        sys.exit(1)

    client = openai.OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=0.1,
    )
    return response.choices[0].message


def run():
    """Main agent loop."""
    tool_defs = load_tool_defs()
    system_prompt, instruction = load_prompts()

    # Build the OpenAI tools format
    tools = []
    for td in tool_defs:
        tools.append({
            "type": "function",
            "function": {
                "name": td["name"],
                "description": td.get("description", ""),
                "parameters": td.get("parameters", {}),
            }
        })

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": instruction},
    ]

    max_turns = 20
    for turn in range(max_turns):
        response = call_llm(messages, tools)

        if response.tool_calls:
            messages.append(response.model_dump())
            for tc in response.tool_calls:
                fn_name = tc.function.name
                try:
                    fn_args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    fn_args = {}

                result = call_tool(fn_name, fn_args)
                log_tool_call(fn_name, fn_args, result)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })
        else:
            # Agent done; parse the final reply into answer.json
            content = response.content or ""
            try:
                answer = json.loads(content)
            except json.JSONDecodeError:
                answer = {"raw_response": content}

            with open(ANSWER_PATH, "w", encoding="utf-8") as f:
                json.dump(answer, f, ensure_ascii=False, indent=2)
            print(f"Agent completed in {turn + 1} turns. Answer written to {ANSWER_PATH}")
            return

    print(f"WARNING: Agent reached max turns ({max_turns}) without final answer",
          file=sys.stderr)


if __name__ == "__main__":
    run()
