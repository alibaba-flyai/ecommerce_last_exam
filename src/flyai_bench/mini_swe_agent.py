#!/usr/bin/env python3
"""Mini-SWE Agent for E-Commerce Last Exam benchmark.

Terminal agent that runs inside the container: exposes only bash execution; the LLM invokes tools via the CLI.
On startup it loads system.md (role + tool docs + workflow) and instruction.md (task description),
injecting them into the prompt so the agent need not read these files itself.

Usage (inside the container):
  python3 /app/mini_swe_agent.py

Environment variables:
  LLM_BASE_URL  — OpenAI-compatible API endpoint
  LLM_API_KEY   — API key
  LLM_MODEL     — model name
  MAX_ITERATIONS — max conversation turns (default: 30)
  COMMAND_TIMEOUT — bash command timeout in seconds (default: 120)
"""
import json
import os
import subprocess
import sys
import time

ANSWER_PATH = "/app/answer.json"
TOOL_CALL_LOG = os.environ.get("TOOL_CALL_LOG", "/app/.tool_calls.jsonl")
MAX_ITERATIONS = int(os.environ.get("MAX_ITERATIONS", "30"))
COMMAND_TIMEOUT = int(os.environ.get("COMMAND_TIMEOUT", "120"))

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:4000")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen3.7-plus")

TERMINAL_TOOL = {
    "type": "function",
    "function": {
        "name": "terminal",
        "description": (
            "Execute a bash command in /app and return stdout+stderr. "
            "Use this to call tools, read files, and perform any shell operations."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Bash command to execute",
                },
            },
            "required": ["command"],
        },
    },
}


def load_file(path, default=""):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return default


def execute_command(command, timeout=None):
    timeout = timeout or COMMAND_TIMEOUT
    try:
        r = subprocess.run(
            ["bash", "-c", command],
            capture_output=True, text=True, timeout=timeout,
            cwd="/app",
        )
        output = r.stdout
        if r.stderr:
            output += "\n[stderr]\n" + r.stderr
        if len(output) > 8000:
            output = output[:4000] + "\n...[truncated]...\n" + output[-4000:]
        return output
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout}s"
    except Exception as e:
        return str(e)


def log_tool_call(command, result_snippet=""):
    entry = {"tool": "terminal", "command": command}
    if result_snippet:
        entry["result_preview"] = result_snippet[:200]
    try:
        with open(TOOL_CALL_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


def run():
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package not installed. Run: pip install openai", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

    system_md = load_file("/app/system.md")
    instruction_md = load_file("/app/instruction.md")

    if not system_md:
        system_md = (
            "You are an AI agent solving a task inside a Docker container.\n"
            "Your working directory is /app.\n"
            "Use the terminal tool to run bash commands.\n"
            "When done, write your structured answer to /app/answer.json "
            "and a human-readable summary (at least 100 words) to /app/answer.md.\n"
            "Do NOT read /app/tests/ or .db files (permission denied)."
        )

    if instruction_md:
        user_msg = instruction_md
    else:
        user_msg = (
            "Read the task files in /app/ (e.g. context.json, instruction.md) "
            "to understand what to do, then solve the task."
        )

    messages = [
        {"role": "system", "content": system_md},
        {"role": "user", "content": user_msg},
    ]

    print(f"Agent started: model={LLM_MODEL}, max_iterations={MAX_ITERATIONS}", flush=True)
    if system_md:
        print(f"  system.md: {len(system_md)} chars loaded", flush=True)
    if instruction_md:
        print(f"  instruction.md: {len(instruction_md)} chars loaded", flush=True)

    for turn in range(MAX_ITERATIONS):
        print(f"\n[turn {turn+1}/{MAX_ITERATIONS}]", flush=True)

        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                tools=[TERMINAL_TOOL],
                tool_choice="auto",
                temperature=0.1,
            )
            msg = response.choices[0].message
        except Exception as e:
            print(f"LLM error: {e}", file=sys.stderr)
            time.sleep(2)
            continue

        if msg.tool_calls:
            messages.append(msg.model_dump())

            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                command = args.get("command", "echo 'no command'")
                print(f"  $ {command[:120]}", flush=True)

                result = execute_command(command)
                log_tool_call(command, result)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

            if os.path.exists(ANSWER_PATH) and os.path.getsize(ANSWER_PATH) > 2:
                print(f"answer.json detected ({os.path.getsize(ANSWER_PATH)} bytes), "
                      "giving LLM one more chance to finalize...", flush=True)
        else:
            content = msg.content or ""
            messages.append({"role": "assistant", "content": content})
            print(f"  [text] {content[:150]}", flush=True)

            if os.path.exists(ANSWER_PATH) and os.path.getsize(ANSWER_PATH) > 2:
                print("Task complete. answer.json exists.", flush=True)
                return

    if os.path.exists(ANSWER_PATH) and os.path.getsize(ANSWER_PATH) > 2:
        print(f"Reached max iterations but answer.json exists.", flush=True)
    else:
        print(f"WARNING: max iterations ({MAX_ITERATIONS}) reached without answer.json",
              file=sys.stderr)


if __name__ == "__main__":
    run()
