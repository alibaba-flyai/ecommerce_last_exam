#!/usr/bin/env python3
"""OpenAI-compatible external Agent example for flyai-bench."""
import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

DEFAULT_MAX_TURNS = 30
TRAJECTORY_OUTPUT = "artifacts/trajectory.jsonl"


def read_context(context_path=None):
    context_path = context_path or os.environ.get("FLYAI_BENCH_CONTEXT")
    if not context_path:
        raise RuntimeError(
            "Agent context path is required via --context or FLYAI_BENCH_CONTEXT")
    return json.loads(Path(context_path).read_text(encoding="utf-8"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--context",
        help="Path to the flyai-bench Agent context manifest",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=DEFAULT_MAX_TURNS,
        help=f"Maximum tool-use turns before final synthesis (default: {DEFAULT_MAX_TURNS})",
    )
    return parser.parse_args(argv)


def prepare_trajectory(context):
    output = context.get("output", {})
    if TRAJECTORY_OUTPUT not in output.get("paths", []):
        return None
    path = Path(output["directory"]) / TRAJECTORY_OUTPUT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    return path


def append_trajectory_event(path, event):
    if path is None:
        return
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")


def request_final_answer(client, model, messages, trajectory_path, turn):
    final_instruction = {
        "role": "user",
        "content": (
            "The tool-use phase is over. Stop calling tools and return the final "
            "structured answer now as one valid JSON object. The host wrapper will "
            "write answer.json and answer.md for you."
        ),
    }
    response = client.chat.completions.create(
        model=model,
        messages=[*messages, final_instruction],
        tool_choice="none",
        temperature=0.1,
    )
    message = response.choices[0].message
    append_trajectory_event(trajectory_path, {
        "turn": turn,
        "event": "final_assistant",
        "message": message.model_dump(exclude_none=True),
    })
    return message.content or ""


def call_tool(tool_server_url, tool_name, params=None):
    url = f"{tool_server_url}/call/{tool_name}"
    data = json.dumps(params or {}, ensure_ascii=False).encode()
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        return {"error": exc.read().decode(), "status": exc.code}


def parse_answer(content):
    text = (content or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1])
            if text.lstrip().startswith("json"):
                text = text.lstrip()[4:].lstrip()
    try:
        return json.loads(text), text
    except json.JSONDecodeError:
        return {"response": text}, text


def write_answers(output_dir, answer, final_text):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    structured_answer = answer
    summary = final_text
    if isinstance(answer, dict) and isinstance(answer.get("answer_json"), dict):
        structured_answer = answer["answer_json"]
        if isinstance(answer.get("answer_md"), str) and answer["answer_md"].strip():
            summary = answer["answer_md"].strip()
    rendered = json.dumps(structured_answer, ensure_ascii=False, indent=2)
    (root / "answer.json").write_text(rendered + "\n", encoding="utf-8")
    summary = summary or rendered
    if summary.lstrip().startswith("#"):
        (root / "answer.md").write_text(summary.rstrip() + "\n", encoding="utf-8")
        return
    markdown = (
        "# Agent Result\n\n"
        "The external Agent completed the task and produced the structured "
        "answer below.\n\n"
        f"{summary}\n"
    )
    (root / "answer.md").write_text(markdown, encoding="utf-8")


def main(argv=None):
    try:
        from openai import OpenAI
    except ImportError:
        print("Install the openai package before running this example", file=sys.stderr)
        return 2

    args = parse_args(argv)
    if args.max_turns <= 0:
        raise ValueError("--max-turns must be greater than zero")
    context = read_context(args.context)
    tool_server_url = context["tools"]["url"]
    tools = context["tools"]["definitions"]
    output_dir = context["output"]["directory"]
    trajectory_path = prepare_trajectory(context)
    llm = context.get("llm", {})
    api_key = os.environ.get(llm.get("api_key_env", "LLM_API_KEY"), "")
    client = OpenAI(base_url=llm.get("base_url") or None, api_key=api_key)

    system_prompt = context["prompts"].get("system") or (
        "You are an Agent solving a tool-use benchmark task. Use the supplied "
        "function tools and return the final task answer as one valid JSON object."
    )
    system_prompt += (
        "\n\nYou are running in external Agent mode. Call the supplied function tools "
        "directly; ignore any instructions to invoke them through bash or write files "
        "yourself. The host wrapper writes the required files. When the task is "
        "complete, stop calling tools and return only the final structured answer as "
        "valid JSON."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": context["prompts"].get("user", "")},
    ]

    for turn in range(args.max_turns):
        try:
            response = client.chat.completions.create(
                model=llm.get("model") or os.environ.get("LLM_MODEL", ""),
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.1,
            )
        except Exception as exc:
            append_trajectory_event(trajectory_path, {
                "turn": turn + 1,
                "event": "llm_error",
                "error": repr(exc),
            })
            print(f"LLM request failed on turn {turn + 1}: {exc}", file=sys.stderr)
            time.sleep(2)
            continue

        message = response.choices[0].message
        append_trajectory_event(trajectory_path, {
            "turn": turn + 1,
            "event": "assistant",
            "message": message.model_dump(exclude_none=True),
        })
        if message.tool_calls:
            messages.append(message.model_dump(exclude_none=True))
            for tool_call in message.tool_calls:
                try:
                    arguments = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}
                result = call_tool(
                    tool_server_url, tool_call.function.name, arguments)
                append_trajectory_event(trajectory_path, {
                    "turn": turn + 1,
                    "event": "tool",
                    "tool_call_id": tool_call.id,
                    "name": tool_call.function.name,
                    "arguments": arguments,
                    "result": result,
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })
            continue

        answer, final_text = parse_answer(message.content)
        write_answers(output_dir, answer, final_text)
        print(f"External Agent completed in {turn + 1} turns")
        return 0

    try:
        final_text = request_final_answer(
            client,
            model=llm.get("model") or os.environ.get("LLM_MODEL", ""),
            messages=messages,
            trajectory_path=trajectory_path,
            turn=args.max_turns + 1,
        )
    except Exception as exc:
        append_trajectory_event(trajectory_path, {
            "turn": args.max_turns + 1,
            "event": "final_error",
            "error": repr(exc),
        })
        print(f"Final answer request failed: {exc}", file=sys.stderr)
        return 1
    if not final_text.strip():
        print("Final answer request returned empty content", file=sys.stderr)
        return 1
    answer, rendered = parse_answer(final_text)
    write_answers(output_dir, answer, rendered)
    print(f"External Agent completed after {args.max_turns} tool turns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
