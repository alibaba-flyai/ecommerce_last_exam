#!/usr/bin/env python3
"""Lightweight tool-proxy server that runs as root and proxies the agent's tool calls.

The agent cannot read trip.db directly, but can call tools via HTTP requests to this service.
This service runs as root, can read trip.db, and executes the original tools under /app/tools/*.

Start: python3 /app/.tool_server.py &
Call:  curl -s localhost:9999/call/search_hotel_list --data '{"city":"Hangzhou"}'
"""
import json
import subprocess
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler

PORT = 9999
TOOLS_DIR = "/app/tools"


class ToolHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        path = self.path.strip("/")
        if not path.startswith("call/"):
            self._respond(400, {"error": "use POST /call/<tool_name>"})
            return

        tool_name = path[5:]  # strip "call/"
        tool_path = f"{TOOLS_DIR}/{tool_name}"

        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len).decode() if content_len else "{}"
        try:
            args = json.loads(body) if body.strip() else {}
        except json.JSONDecodeError:
            self._respond(400, {"error": "invalid JSON body"})
            return

        cmd = [tool_path]
        for k, v in args.items():
            flag = f"--{k.replace('_', '-')}"
            if isinstance(v, (list, dict)):
                cmd += [flag, json.dumps(v, ensure_ascii=False)]
            else:
                cmd += [flag, str(v)]

        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if r.returncode != 0:
                self._respond(500, {"error": r.stderr[:1000], "returncode": r.returncode})
            else:
                try:
                    result = json.loads(r.stdout)
                    self._respond(200, result)
                except json.JSONDecodeError:
                    self._respond(200, {"raw": r.stdout[:5000]})
        except FileNotFoundError:
            self._respond(404, {"error": f"tool not found: {tool_name}"})
        except subprocess.TimeoutExpired:
            self._respond(504, {"error": "tool timeout"})

    def do_GET(self):
        if self.path == "/health":
            self._respond(200, {"status": "ok"})
        elif self.path == "/tools":
            import os
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
    server = HTTPServer(("127.0.0.1", PORT), ToolHandler)
    print(f"Tool server listening on 127.0.0.1:{PORT}", file=sys.stderr)
    server.serve_forever()
