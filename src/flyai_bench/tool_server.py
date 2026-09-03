#!/usr/bin/env python3
"""Lightweight tool-proxy server that runs as root and proxies the agent's tool calls.

The agent cannot read trip.db directly, but can call tools via HTTP requests to this service.
This service runs as root, can read trip.db, and executes the original tools under /app/tools/*.

Start: python3 /app/.tool_server.py &
Call:  curl -s localhost:9999/call/search_hotel_list --data '{"city":"Hangzhou"}'
"""
import json
import os
import re
import subprocess
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler

PORT = 9999
TOOLS_DIR = "/app/tools"
TOOL_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


def validate_tool_name(tool_name):
    """Reject paths and hidden files while allowing schema-defined tool names."""
    if not isinstance(tool_name, str) or not TOOL_NAME_PATTERN.fullmatch(tool_name):
        raise ValueError(f"invalid tool name: {tool_name!r}")
    return tool_name


def build_tool_command(tool_path, arguments):
    """Render JSON tool arguments without changing schema property names."""
    command = [tool_path]
    for key, value in arguments.items():
        if not isinstance(key, str) or not key or key.startswith("-"):
            raise ValueError(f"invalid tool argument name: {key!r}")
        command.append(f"--{key}")
        if isinstance(value, (list, dict, bool)) or value is None:
            command.append(json.dumps(value, ensure_ascii=False))
        else:
            command.append(str(value))
    return command


class ToolHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        path = self.path.strip("/")
        if not path.startswith("call/"):
            self._respond(400, {"error": "use POST /call/<tool_name>"})
            return

        tool_name = path[5:]  # strip "call/"
        try:
            validate_tool_name(tool_name)
        except ValueError as exc:
            self._respond(400, {"error": str(exc)})
            return
        tool_path = f"{TOOLS_DIR}/{tool_name}"

        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len).decode() if content_len else "{}"
        try:
            args = json.loads(body) if body.strip() else {}
        except json.JSONDecodeError:
            self._respond(400, {"error": "invalid JSON body"})
            return
        if not isinstance(args, dict):
            self._respond(400, {"error": "JSON body must be an object"})
            return

        try:
            cmd = build_tool_command(tool_path, args)
        except ValueError as exc:
            self._respond(400, {"error": str(exc)})
            return

        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if r.returncode != 0:
                payload = {"error": r.stderr[:1000], "returncode": r.returncode}
                self._respond(500, payload)
            else:
                try:
                    payload = json.loads(r.stdout)
                except json.JSONDecodeError:
                    payload = {"raw": r.stdout[:5000]}
                self._respond(200, payload)
        except FileNotFoundError:
            payload = {"error": f"tool not found: {tool_name}"}
            self._respond(404, payload)
        except subprocess.TimeoutExpired:
            payload = {"error": "tool timeout"}
            self._respond(504, payload)

    def do_GET(self):
        if self.path == "/health":
            self._respond(200, {"status": "ok"})
        elif self.path == "/tools":
            tools = [f for f in os.listdir(TOOLS_DIR)
                     if os.path.isfile(f"{TOOLS_DIR}/{f}") and not f.endswith(".py")]
            self._respond(200, {"tools": tools})
        else:
            self._respond(404, {"error": "not found"})

    def _respond(self, code, data):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass  # suppress access logs


if __name__ == "__main__":
    bind = "0.0.0.0" if "--expose" in sys.argv else "127.0.0.1"
    server = HTTPServer((bind, PORT), ToolHandler)
    print(f"Tool server listening on {bind}:{PORT}", file=sys.stderr)
    server.serve_forever()
