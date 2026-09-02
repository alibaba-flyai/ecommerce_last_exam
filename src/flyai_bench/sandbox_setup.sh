#!/bin/bash
# Runs inside the container: set up permission isolation + start the Tool Server
# Injected by run_eval.py and executed as root
# Auto-adapts to two layouts:
#   trip_bench:        /app/tools/        + /app/trip.db
#   omni_consume_en:   /app/environment/tools/ + /app/environment/consume_mix.db

set -e

# -- Auto-detect the tools directory --
if [ -d /app/tools ] && [ ! -d /app/environment/tools ]; then
    TOOLS_DIR="/app/tools"
elif [ -d /app/environment/tools ]; then
    TOOLS_DIR="/app/environment/tools"
else
    echo "ERROR: no tools directory found" >&2
    exit 1
fi
echo "tools layout: $TOOLS_DIR"

# 1. Create the agent user
useradd -m -s /bin/bash agent 2>/dev/null || true

# 2. tests/ accessible only by root (the image may not contain tests/)
if [ -d /app/tests ]; then
    chmod 700 /app/tests
    chown -R root:root /app/tests
fi

# 3. All .db files readable only by root (trip.db, consume_mix.db, common.db, etc.)
find /app -name '*.db' -exec chmod 600 {} + 2>/dev/null || true
find /app -name '*.db' -exec chown root:root {} + 2>/dev/null || true

# 4. Back up the original tools; tool_server calls the real ones
cp -a "$TOOLS_DIR" /app/.tools_real

# 5. Start the Tool Server (root, background, calls .tools_real/)
sed -i 's|TOOLS_DIR = "/app/tools"|TOOLS_DIR = "/app/.tools_real"|' /app/.tool_server.py
if [ "${AGENT_MODE}" = "external" ]; then
    python3 /app/.tool_server.py --expose &
else
    python3 /app/.tool_server.py &
fi
sleep 0.3

# 6. Write the Python tool wrapper (correctly handles JSON array/object arguments)
cat > /app/.tool_wrapper.py << 'PYWRAPPER'
import sys, json, urllib.request

tool_name = sys.argv[1]
args = sys.argv[2:]
params = {}
i = 0
while i < len(args):
    if args[i].startswith("--"):
        key = args[i][2:].replace("-", "_")
        if i + 1 < len(args):
            val = args[i + 1]
            try:
                params[key] = json.loads(val)
            except (json.JSONDecodeError, ValueError):
                params[key] = val
            i += 2
        else:
            i += 1
    else:
        i += 1

data = json.dumps(params, ensure_ascii=False).encode()
req = urllib.request.Request(
    f"http://127.0.0.1:9999/call/{tool_name}",
    data=data,
    headers={"Content-Type": "application/json"},
)
try:
    with urllib.request.urlopen(req, timeout=120) as resp:
        sys.stdout.write(resp.read().decode())
except urllib.error.HTTPError as e:
    sys.stdout.write(e.read().decode())
    sys.exit(1)
except Exception as e:
    sys.stdout.write(json.dumps({"error": str(e)}))
    sys.exit(1)
PYWRAPPER
chmod 644 /app/.tool_wrapper.py

# 7. Replace each tool in the tools dir with the Python wrapper (supports 'python3 <tool>' and direct execution)
for tool_path in /app/.tools_real/*; do
    name=$(basename "$tool_path")
    if [[ "$name" == *.py ]]; then
        continue
    fi
    cat > "$TOOLS_DIR/$name" << WRAPPER
#!/usr/bin/env python3
import sys, os
os.execvp("python3", ["python3", "/app/.tool_wrapper.py", "$name"] + sys.argv[1:])
WRAPPER
    chmod 755 "$TOOLS_DIR/$name"
done

# 8. Files readable by the agent
chmod o+r /app/tool_defs.json 2>/dev/null || true
chmod o+r /app/system.md /app/instruction.md 2>/dev/null || true
chmod o+r /app/context.json /app/environment/context.json 2>/dev/null || true
chmod -R o+rx "$TOOLS_DIR"

# 9. Output files and log dirs writable by the agent
touch /app/answer.json /app/answer.md /app/.tool_calls.jsonl
chown agent:agent /app/answer.json /app/answer.md /app/.tool_calls.jsonl
mkdir -p /app/logs
chown agent:agent /app/logs

# 10. .tools_real/ and .tool_server.py are inaccessible to the agent
chmod 700 /app/.tools_real
chmod 600 /app/.tool_server.py

echo "sandbox ready"
