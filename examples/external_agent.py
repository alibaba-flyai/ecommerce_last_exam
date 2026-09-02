#!/usr/bin/env python3
"""Example external agent for flyai-bench.

This script demonstrates the external agent protocol. It reads task prompts
from environment variables, calls tools via the Tool Server HTTP API, and
writes answer.json to OUTPUT_DIR.

Usage:
  flyai-bench run --agent-mode external --agent-cmd "python3 examples/external_agent.py" --limit 1
"""
import json
import os
import sys
import urllib.request

TOOL_SERVER_URL = os.environ["TOOL_SERVER_URL"]
OUTPUT_DIR = os.environ["OUTPUT_DIR"]
INSTANCE_ID = os.environ.get("INSTANCE_ID", "unknown")


def read_file(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def call_tool(tool_name, params=None):
    url = f"{TOOL_SERVER_URL}/call/{tool_name}"
    data = json.dumps(params or {}, ensure_ascii=False).encode()
    req = urllib.request.Request(url, data=data,
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


def list_tools():
    url = f"{TOOL_SERVER_URL}/tools"
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read().decode())


def main():
    system_prompt = read_file(os.environ.get("SYSTEM_PROMPT_PATH", ""))
    instruction = read_file(os.environ.get("INSTRUCTION_PATH", ""))
    tool_defs_path = os.environ.get("TOOL_DEFS_PATH", "")
    tool_defs = json.loads(read_file(tool_defs_path)) if tool_defs_path else []

    print(f"[external_agent] instance={INSTANCE_ID}")
    print(f"[external_agent] tool_server={TOOL_SERVER_URL}")
    print(f"[external_agent] system_prompt={len(system_prompt)} chars")
    print(f"[external_agent] instruction={len(instruction)} chars")
    print(f"[external_agent] tool_defs={len(tool_defs)} tools")

    tools = list_tools()
    print(f"[external_agent] available tools: {tools.get('tools', [])}")

    # --- Replace the logic below with your own agent ---

    # Example: call the first available tool with no arguments
    available = tools.get("tools", [])
    tool_result = None
    if available:
        try:
            tool_result = call_tool(available[0])
            print(f"[external_agent] called {available[0]}, got {str(tool_result)[:200]}")
        except Exception as e:
            print(f"[external_agent] tool call failed: {e}", file=sys.stderr)

    # Write a placeholder answer
    answer = {
        "note": "This is an example external agent. Replace with your own logic.",
        "instance_id": INSTANCE_ID,
        "tool_result_sample": tool_result,
    }
    answer_path = os.path.join(OUTPUT_DIR, "answer.json")
    with open(answer_path, "w", encoding="utf-8") as f:
        json.dump(answer, f, ensure_ascii=False, indent=2)
    print(f"[external_agent] wrote {answer_path}")


if __name__ == "__main__":
    main()
