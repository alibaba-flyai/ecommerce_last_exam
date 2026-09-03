#!/usr/bin/env python3
"""E-Commerce Last Exam evaluation CLI.

Usage:
  flyai-bench run                                    # run with default config
  flyai-bench --config eval_config.local.yaml run    # custom config
  flyai-bench run --dataset-config e_commerce --limit 10
  flyai-bench run --dry-run
  flyai-bench status
  flyai-bench report
"""
import argparse
import copy
import json
import os
import posixpath
import re
import shlex
import string
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

try:
    import yaml
except ImportError:
    yaml = None


# ─── Config ───────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "dataset": {
        "repo_id": "FlyaiLab/ecommerce_last_exam",
        "config": "travel",
        "split": "test",
    },
    "docker": {
        "registry": "",
        "platform": "linux/amd64",
    },
    "agent": {
        "cmd": "python3 /app/mini_swe_agent.py",
        "output_paths": [],
        "timeout_sec": 1800,
        "llm_base_url": "http://host.docker.internal:4000",
        "llm_api_key": "",
        "llm_model": "qwen3.7-plus",
    },
    "verifier": {
        "timeout_sec": 1200,
        "judge_base_url": "http://host.docker.internal:4000",
        "judge_api_key": "",
        "judge_model": "qwen3.7-plus",
    },
    "runner": {
        "concurrency": 1,
        "output_dir": "./eval_results",
        "limit": None,
        "skip_done": True,
    },
}

REQUIRED_EXTERNAL_OUTPUTS = ("answer.json", "answer.md")
REGISTERED_AGENT_PROTOCOL = "manifest-v1"
REGISTERED_AGENT_ENV_BASE = (
    "PATH", "HOME", "USER", "LOGNAME", "TMPDIR", "TMP", "TEMP",
    "LANG", "LC_ALL", "SYSTEMROOT", "WINDIR",
)
ENV_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_external_output_path(value):
    if not isinstance(value, str) or not value or "\\" in value or "\0" in value:
        raise ValueError(f"invalid external output path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or path == PurePosixPath(".") or ".." in path.parts:
        raise ValueError(f"external output path must stay below OUTPUT_DIR: {value!r}")
    normalized = path.as_posix()
    if normalized not in REQUIRED_EXTERNAL_OUTPUTS and path.parts[0] != "artifacts":
        raise ValueError(
            "additional external outputs must be placed below artifacts/: "
            f"{value!r}")
    return normalized


def external_output_paths(agent_cfg):
    """Return required answer files followed by configured additional outputs."""
    configured = agent_cfg.get("output_paths")
    if not configured and agent_cfg.get("output_path"):
        configured = agent_cfg.get("output_path", [])
    elif configured is None:
        configured = []
    if isinstance(configured, str):
        configured = [configured]
    if not isinstance(configured, (list, tuple)):
        raise ValueError("agent.output_paths must be a string or a list of strings")

    result = list(REQUIRED_EXTERNAL_OUTPUTS)
    for value in configured:
        normalized = _validate_external_output_path(value)
        if normalized not in result:
            result.append(normalized)
    return result


def _external_output_source(output_dir, relative_path):
    base = Path(output_dir).resolve()
    source = base.joinpath(*PurePosixPath(relative_path).parts)
    try:
        resolved = source.resolve()
    except OSError:
        return None
    if not resolved.is_relative_to(base) or not resolved.is_file():
        return None
    return source


def missing_external_outputs(output_dir, output_paths):
    return [
        relative_path
        for relative_path in output_paths
        if _external_output_source(output_dir, relative_path) is None
    ]


def clear_external_outputs(output_dir, output_paths):
    """Remove declared files left by an earlier failed or repeated run."""
    base = Path(output_dir).resolve()
    for relative_path in output_paths:
        path = base.joinpath(*PurePosixPath(relative_path).parts)
        if not path.resolve().is_relative_to(base):
            raise ValueError(
                f"external output path resolves outside OUTPUT_DIR: {relative_path!r}")
        if path.is_symlink() or path.is_file():
            path.unlink()


def _read_optional_text(path):
    try:
        return Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def _read_optional_json(path, default):
    text = _read_optional_text(path)
    return json.loads(text) if text else default


def _read_external_tool_definitions(prompt_root):
    path = prompt_root / "tool_defs.json"
    try:
        definitions = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(
            "external Agent mode requires /app/tool_defs.json in the task image") from exc
    if not isinstance(definitions, list) or not definitions:
        raise ValueError(
            "external Agent mode requires a non-empty /app/tool_defs.json in the task image")
    return definitions


def write_external_agent_context(cfg, instance_id, tool_server_url, output_dir,
                                 prompt_dir, output_paths):
    """Write the complete non-secret task contract for an external Agent."""
    prompt_root = Path(prompt_dir).resolve()
    context_path = prompt_root / "agent_context.json"
    agent_cfg = cfg["agent"]
    context = {
        "protocol_version": "1.0",
        "instance_id": instance_id,
        "prompts": {
            "system": _read_optional_text(prompt_root / "system.md"),
            "user": _read_optional_text(prompt_root / "instruction.md"),
        },
        "tools": {
            "url": tool_server_url,
            "definitions": _read_external_tool_definitions(prompt_root),
        },
        "task_context": _read_optional_json(prompt_root / "context.json", None),
        "output": {
            "directory": str(Path(output_dir).resolve()),
            "paths": output_paths,
        },
        "llm": {
            "base_url": agent_cfg.get("llm_base_url", ""),
            "model": agent_cfg.get("llm_model", ""),
            "api_key_env": "LLM_API_KEY",
        },
    }
    context_path.write_text(
        json.dumps(context, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(context_path)


def _registered_agent_command_args(command):
    if isinstance(command, str):
        try:
            command = shlex.split(command)
        except ValueError as exc:
            raise ValueError(f"invalid Agent command: {exc}") from exc
    elif isinstance(command, (list, tuple)):
        command = list(command)
    else:
        raise ValueError("Agent command must be a string or a list of strings")
    if not command or any(not isinstance(arg, str) for arg in command):
        raise ValueError("Agent command must contain one or more string arguments")
    if not command[0]:
        raise ValueError("Agent command executable must not be empty")
    return command


def render_registered_agent_command(command, placeholders):
    """Render a registered Agent command into an argv list."""
    formatter = string.Formatter()
    rendered = []
    for arg in _registered_agent_command_args(command):
        try:
            fields = list(formatter.parse(arg))
        except ValueError as exc:
            raise ValueError(f"invalid Agent command template {arg!r}: {exc}") from exc
        for _, field_name, format_spec, conversion in fields:
            if field_name is None:
                continue
            if field_name not in placeholders:
                raise ValueError(
                    f"unknown Agent command placeholder {field_name!r} in {arg!r}")
            if format_spec or conversion:
                raise ValueError(
                    f"Agent command placeholder {field_name!r} cannot use formatting")
        rendered.append(arg.format_map(placeholders))
    return rendered


def agent_command_placeholders(instance_id, tool_server_url, output_dir,
                               prompt_dir, agent_context_path, config_dir):
    prompt_root = Path(prompt_dir).resolve()
    return {
        "context": str(Path(agent_context_path).resolve()),
        "output_dir": str(Path(output_dir).resolve()),
        "instance_id": instance_id,
        "tool_server_url": tool_server_url,
        "system_prompt_path": str(prompt_root / "system.md"),
        "instruction_path": str(prompt_root / "instruction.md"),
        "tool_defs_path": str(prompt_root / "tool_defs.json"),
        "task_context_path": str(prompt_root / "context.json"),
        "config_dir": str(Path(config_dir).resolve()),
    }


def resolve_registered_agent_cwd(registration, config_dir):
    cwd = registration.get("cwd", ".")
    if not isinstance(cwd, str) or not cwd:
        raise ValueError("registered Agent cwd must be a non-empty string")
    path = Path(cwd).expanduser()
    if not path.is_absolute():
        path = Path(config_dir) / path
    path = path.resolve()
    if not path.is_dir():
        raise ValueError(f"registered Agent working directory does not exist: {path}")
    return str(path)


def build_registered_agent_env(cfg, registration, base_env=None):
    """Build the minimal host environment for a registered Agent."""
    source = dict(os.environ if base_env is None else base_env)
    env = {name: source[name] for name in REGISTERED_AGENT_ENV_BASE if name in source}
    pass_env = registration.get("pass_env", [])
    if not isinstance(pass_env, (list, tuple)):
        raise ValueError("registered Agent pass_env must be a list of variable names")
    for name in pass_env:
        if not isinstance(name, str) or not ENV_NAME_PATTERN.fullmatch(name):
            raise ValueError(f"invalid registered Agent pass_env name: {name!r}")
        if name in source:
            env[name] = source[name]

    agent_cfg = cfg.get("agent", {})
    llm_api_key = agent_cfg.get("llm_api_key") or source.get("LLM_API_KEY", "")
    env.update({
        "LLM_BASE_URL": str(agent_cfg.get("llm_base_url", "")),
        "LLM_API_KEY": str(llm_api_key),
        "LLM_MODEL": str(agent_cfg.get("llm_model", "")),
    })
    return env


def run_registered_agent(cfg, registration, placeholders, base_env=None):
    command = render_registered_agent_command(registration.get("command"), placeholders)
    cwd = resolve_registered_agent_cwd(
        registration, cfg.get("_config_dir", os.getcwd()))
    env = build_registered_agent_env(cfg, registration, base_env=base_env)
    timeout = registration.get("timeout_sec", cfg["agent"]["timeout_sec"])
    return subprocess.run(
        command,
        shell=False,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def launch_external_agent(cfg, instance_id, tool_server_url, output_dir,
                          prompt_dir, output_paths, agent_context_path):
    """Launch the selected registered Agent."""
    registration = cfg["agent"].get("registration")
    if not registration:
        raise ValueError("registered Agent is not selected")
    placeholders = agent_command_placeholders(
        instance_id=instance_id,
        tool_server_url=tool_server_url,
        output_dir=output_dir,
        prompt_dir=prompt_dir,
        agent_context_path=agent_context_path,
        config_dir=cfg.get("_config_dir", os.getcwd()),
    )
    return run_registered_agent(cfg, registration, placeholders)


def copy_external_outputs(container_name, output_dir, output_paths):
    missing = missing_external_outputs(output_dir, output_paths)
    if missing:
        return False, f"required output files not produced: {', '.join(missing)}"

    for relative_path in output_paths:
        source = _external_output_source(output_dir, relative_path)
        container_path = f"/app/{relative_path}"
        parent = posixpath.dirname(container_path)
        if parent != "/app":
            code, _, stderr = docker_exec(
                container_name, f"mkdir -p -- {shlex.quote(parent)}", timeout=10)
            if code != 0:
                return False, f"failed to create {parent}: {stderr[-200:]}"
        result = subprocess.run(
            ["docker", "cp", str(source), f"{container_name}:{container_path}"],
            capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return False, f"failed to copy {relative_path}: {result.stderr[-200:]}"
    return True, ""


def external_tool_port_publish_spec():
    """Publish the ephemeral Tool Server port on the host loopback only."""
    return "127.0.0.1::9999"


def load_config(path):
    config_dir = str(Path(path).expanduser().resolve().parent) if path else os.getcwd()
    if not path or not os.path.exists(path):
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        cfg["_config_dir"] = config_dir
        return cfg
    if yaml is None:
        print("WARNING: pyyaml not installed, using default config", file=sys.stderr)
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        cfg["_config_dir"] = config_dir
        return cfg
    with open(path, encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    if not isinstance(user_cfg, dict):
        raise ValueError("config file must contain a YAML mapping")
    user_agent_cfg = user_cfg.get("agent", {})
    if not isinstance(user_agent_cfg, dict):
        raise ValueError("agent must be a mapping")
    removed_fields = sorted({"mode", "external_cmd"} & user_agent_cfg.keys())
    if removed_fields:
        fields = ", ".join(removed_fields)
        raise ValueError(
            f"unsupported agent field(s): {fields}; register custom Agents under "
            "agents and select one with agent.name or --agent")
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    for section in DEFAULT_CONFIG:
        if section in user_cfg and isinstance(user_cfg[section], dict):
            cfg[section] = {**cfg[section], **user_cfg[section]}
    agents = user_cfg.get("agents", {})
    if not isinstance(agents, dict):
        raise ValueError("agents must be a mapping of Agent names to registrations")
    cfg["agents"] = copy.deepcopy(agents)
    cfg["_config_dir"] = config_dir
    return cfg


def resolve_registered_agent(cfg, selected_name=None):
    """Activate and return a named Agent registration, if one is selected."""
    agent_cfg = cfg.setdefault("agent", {})
    name = selected_name or agent_cfg.get("name")
    if not name:
        return None
    if not isinstance(name, str):
        raise ValueError("agent.name must be a string")

    agents = cfg.get("agents", {})
    if not isinstance(agents, dict):
        raise ValueError("agents must be a mapping of Agent names to registrations")
    raw_registration = agents.get(name)
    if not isinstance(raw_registration, dict):
        known = ", ".join(sorted(agents)) or "none"
        raise ValueError(f"unknown Agent {name!r}; registered Agents: {known}")

    registration = copy.deepcopy(raw_registration)
    protocol = registration.get("protocol", REGISTERED_AGENT_PROTOCOL)
    if protocol != REGISTERED_AGENT_PROTOCOL:
        raise ValueError(
            f"registered Agent {name!r} uses unsupported protocol {protocol!r}; "
            f"expected {REGISTERED_AGENT_PROTOCOL!r}")
    _registered_agent_command_args(registration.get("command"))
    timeout_sec = registration.get("timeout_sec")
    if timeout_sec is not None and (
            isinstance(timeout_sec, bool)
            or not isinstance(timeout_sec, (int, float))
            or timeout_sec <= 0):
        raise ValueError(
            f"registered Agent {name!r} timeout_sec must be a positive number")
    registration["name"] = name
    registration["protocol"] = protocol

    agent_cfg["name"] = name
    if "timeout_sec" in registration:
        agent_cfg["timeout_sec"] = registration["timeout_sec"]
    if "output_paths" in registration:
        agent_cfg["output_paths"] = copy.deepcopy(registration["output_paths"])
    agent_cfg["registration"] = registration
    return registration


def apply_overrides(cfg, args):
    if getattr(args, "dataset_config", None):
        cfg["dataset"]["config"] = args.dataset_config
    if getattr(args, "limit", None) is not None:
        cfg["runner"]["limit"] = args.limit
    if getattr(args, "concurrency", None) is not None:
        cfg["runner"]["concurrency"] = args.concurrency
    if getattr(args, "output_dir", None):
        cfg["runner"]["output_dir"] = args.output_dir
    if getattr(args, "registry", None):
        cfg["docker"]["registry"] = args.registry
    resolve_registered_agent(cfg, getattr(args, "agent", None))
    return cfg


# ─── Dataset ──────────────────────────────────────────────────────────────────

def load_instances(cfg):
    repo_id = cfg["dataset"]["repo_id"]
    config = cfg["dataset"]["config"]
    split = cfg["dataset"]["split"]
    try:
        from datasets import load_dataset
        ds = load_dataset(repo_id, config, split=split)
        return [dict(row) for row in ds]
    except ImportError:
        pass
    try:
        import pandas as pd
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id, f"data/{config}/{split}-00000-of-00001.parquet",
                               repo_type="dataset")
        df = pd.read_parquet(path)
        return df.to_dict(orient="records")
    except Exception:
        pass
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id, f"data/{config}/{split}.jsonl", repo_type="dataset")
    instances = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                instances.append(json.loads(line))
    return instances


# ─── Docker helpers ───────────────────────────────────────────────────────────

def resolve_image(docker_image, registry):
    if registry and "/" in docker_image and not docker_image.startswith(registry):
        return f"{registry}/{docker_image}"
    return docker_image


def docker_pull(image, timeout=300):
    r = subprocess.run(["docker", "pull", image],
                       capture_output=True, text=True, timeout=timeout)
    return r.returncode == 0, r.stderr[-300:] if r.returncode != 0 else ""


def docker_run(image, name, platform, env_vars=None, timeout_sleep=3600):
    cmd = ["docker", "run", "-d", "--name", name, "--platform", platform]
    if env_vars:
        for k, v in env_vars.items():
            if v:
                cmd += ["-e", f"{k}={v}"]
    cmd += [image, "sleep", str(timeout_sleep)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return r.returncode == 0, r.stdout.strip(), r.stderr[-200:]


def docker_exec(name, cmd_str, timeout=1800):
    cmd = ["docker", "exec", name, "bash", "-c", cmd_str]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr


def docker_cp_from(name, container_path, local_path):
    r = subprocess.run(["docker", "cp", f"{name}:{container_path}", local_path],
                       capture_output=True, text=True, timeout=30)
    return r.returncode == 0


def docker_rm(name):
    subprocess.run(["docker", "rm", "-f", name],
                   capture_output=True, text=True, timeout=30)


def docker_port(name, container_port=9999):
    r = subprocess.run(["docker", "port", name, str(container_port)],
                       capture_output=True, text=True, timeout=10)
    if r.returncode != 0:
        return None
    mapping = r.stdout.strip().split("\n")[0]
    return mapping.split(":")[-1]


# ─── Evaluation logic ─────────────────────────────────────────────────────────

def eval_instance(instance, cfg, run_id=None):
    """Evaluate a single instance: one container, agent runs as non-root (cannot read tests/ or trip.db)."""
    instance_id = instance["instance_id"]
    registry = cfg["docker"]["registry"]
    platform = cfg["docker"]["platform"]
    if run_id is not None:
        output_dir = os.path.join(cfg["runner"]["output_dir"], cfg["dataset"]["config"],
                                  instance_id, f"run_{run_id}")
    else:
        output_dir = os.path.join(cfg["runner"]["output_dir"], cfg["dataset"]["config"], instance_id)
    os.makedirs(output_dir, exist_ok=True)

    result = {"instance_id": instance_id, "domain": instance.get("domain", "")}
    if run_id is not None:
        result["run_id"] = run_id
    t0 = time.time()
    suffix = f"_r{run_id}" if run_id is not None else ""
    container_name = f"eval_{instance_id[:12]}_{int(t0) % 10000}{suffix}"
    container_started = False

    is_external = bool(cfg["agent"].get("registration"))

    try:
        # 1. Pull
        image = resolve_image(instance["docker_image"], registry)
        ok, err = docker_pull(image)
        if not ok:
            result.update(reward=None, error=f"pull failed: {err}", phase="pull")
            return result

        # 2. Start the container (root) and set up permission isolation
        all_env = {
            "LLM_BASE_URL": cfg["agent"]["llm_base_url"],
            "LLM_API_KEY": cfg["agent"]["llm_api_key"],
            "LLM_MODEL": cfg["agent"]["llm_model"],
            "JUDGE_BASE_URL": cfg["verifier"]["judge_base_url"],
            "JUDGE_API_KEY": cfg["verifier"]["judge_api_key"],
            "JUDGE_MODEL": cfg["verifier"]["judge_model"],
        }
        if is_external:
            all_env["AGENT_MODE"] = "external"
        run_cmd = ["docker", "run", "-d", "--name", container_name, "--platform", platform]
        if is_external:
            run_cmd += ["-p", external_tool_port_publish_spec()]
        for k, v in all_env.items():
            if v:
                run_cmd += ["-e", f"{k}={v}"]
        run_cmd += [image, "sleep", str(cfg["agent"]["timeout_sec"] + cfg["verifier"]["timeout_sec"] + 120)]
        r = subprocess.run(run_cmd, capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            result.update(reward=None, error=f"start failed: {r.stderr[-200:]}", phase="start")
            return result
        container_started = True

        # 3. Inject sandbox script + tool_server + agent, set up permission isolation
        script_dir = os.path.dirname(os.path.abspath(__file__))
        subprocess.run(["docker", "cp", os.path.join(script_dir, "tool_server.py"),
                        f"{container_name}:/app/.tool_server.py"], capture_output=True, timeout=10)
        subprocess.run(["docker", "cp", os.path.join(script_dir, "sandbox_setup.sh"),
                        f"{container_name}:/app/.sandbox_setup.sh"], capture_output=True, timeout=10)
        subprocess.run(["docker", "cp", os.path.join(script_dir, "mini_swe_agent.py"),
                        f"{container_name}:/app/mini_swe_agent.py"], capture_output=True, timeout=10)
        code, _, stderr = docker_exec(container_name, "bash /app/.sandbox_setup.sh", timeout=30)
        if code != 0:
            result.update(reward=None, error=f"sandbox setup failed: {stderr[-200:]}", phase="setup")
            return result

        if is_external:
            # ── External agent mode ──
            try:
                output_paths = external_output_paths(cfg["agent"])
            except ValueError as exc:
                result.update(reward=None, error=str(exc), phase="config")
                return result

            # 4a. Extract prompt files from container to OUTPUT_DIR/.prompt/
            prompt_dir = os.path.join(output_dir, ".prompt")
            os.makedirs(prompt_dir, exist_ok=True)
            for fname in ["system.md", "instruction.md", "tool_defs.json", "context.json"]:
                for container_path in [f"/app/{fname}", f"/app/environment/{fname}"]:
                    if docker_cp_from(container_name, container_path, os.path.join(prompt_dir, fname)):
                        break

            # 4b. Get the mapped host port for tool_server
            host_port = docker_port(container_name, 9999)
            if not host_port:
                result.update(reward=None, error="failed to get mapped port for tool_server", phase="setup")
                return result

            # 4c. Run external agent command on the host
            clear_external_outputs(output_dir, output_paths)
            agent_context_path = write_external_agent_context(
                cfg,
                instance_id=instance_id,
                tool_server_url=f"http://localhost:{host_port}",
                output_dir=output_dir,
                prompt_dir=prompt_dir,
                output_paths=output_paths,
            )
            r = launch_external_agent(
                cfg,
                instance_id=instance_id,
                tool_server_url=f"http://localhost:{host_port}",
                output_dir=output_dir,
                prompt_dir=prompt_dir,
                output_paths=output_paths,
                agent_context_path=agent_context_path,
            )
            with open(os.path.join(output_dir, "agent_stdout.txt"), "w") as f:
                f.write(r.stdout)
            with open(os.path.join(output_dir, "agent_stderr.txt"), "w") as f:
                f.write(r.stderr)

            if r.returncode != 0:
                result.update(reward=None, error=f"agent exit {r.returncode}: {r.stderr[-100:]}",
                              phase="agent")
                return result

            # 4d. Copy every declared Agent output from the host into /app
            copied, copy_error = copy_external_outputs(
                container_name, output_dir, output_paths)
            if not copied:
                result.update(reward=None, error=copy_error, phase="agent")
                return result
            docker_cp_from(container_name, "/app/.tool_calls.jsonl",
                           os.path.join(output_dir, "tool_calls.jsonl"))

        else:
            # ── Internal agent mode (existing behavior) ──
            # 4. Run the agent as the agent user (cannot read tests/ or trip.db)
            agent_cmd = cfg["agent"]["cmd"]
            code, stdout, stderr = docker_exec(
                container_name, f"su agent -c '{agent_cmd}'",
                timeout=cfg["agent"]["timeout_sec"])
            with open(os.path.join(output_dir, "agent_stdout.txt"), "w") as f:
                f.write(stdout)
            with open(os.path.join(output_dir, "agent_stderr.txt"), "w") as f:
                f.write(stderr)

            if code != 0:
                result.update(reward=None, error=f"agent exit {code}: {stderr[-100:]}",
                              phase="agent")
                return result

            # Copy agent output to the host
            docker_cp_from(container_name, "/app/answer.json", os.path.join(output_dir, "answer.json"))
            docker_cp_from(container_name, "/app/.tool_calls.jsonl",
                           os.path.join(output_dir, "tool_calls.jsonl"))

            if not os.path.exists(os.path.join(output_dir, "answer.json")):
                result.update(reward=None, error="answer.json not produced", phase="agent")
                return result

        # 5. Run the verifier as root (tests/ and trip.db readable)
        code, stdout, stderr = docker_exec(
            container_name, "cd /app && TESTS_DIR=/app/tests bash tests/test.sh",
            timeout=cfg["verifier"]["timeout_sec"])
        with open(os.path.join(output_dir, "verifier_stdout.txt"), "w") as f:
            f.write(stdout)
        with open(os.path.join(output_dir, "verifier_stderr.txt"), "w") as f:
            f.write(stderr)

        # 6. Extract reward
        code, reward_str, _ = docker_exec(
            container_name,
            "cat /app/reward.txt 2>/dev/null || cat /logs/verifier/reward.txt 2>/dev/null",
            timeout=10)
        if code == 0 and reward_str.strip():
            try:
                reward = float(reward_str.strip())
                result.update(reward=reward, error=None)
            except ValueError:
                result.update(reward=None, error=f"invalid reward: {reward_str.strip()}",
                              phase="verifier")
        else:
            result.update(reward=None, error="reward.txt not found", phase="verifier")

    except subprocess.TimeoutExpired as e:
        result.update(reward=None, error=f"timeout: {e}", phase="timeout")
    except Exception as e:
        result.update(reward=None, error=repr(e)[:200], phase="exception")
    finally:
        if container_started:
            docker_cp_from(container_name, "/app/.tool_calls.jsonl",
                           os.path.join(output_dir, "tool_calls.jsonl"))
            docker_rm(container_name)
        result["duration_sec"] = round(time.time() - t0, 1)
        with open(os.path.join(output_dir, "result.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    return result


# ─── Commands ─────────────────────────────────────────────────────────────────

def cmd_run(args):
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)

    print(f"Loading dataset: {cfg['dataset']['repo_id']} config={cfg['dataset']['config']}")
    instances = load_instances(cfg)

    limit = cfg["runner"]["limit"]
    if limit:
        instances = instances[:limit]

    runs = getattr(args, "runs", 1) or 1
    use_run_id = runs > 1

    if args.dry_run:
        print(f"Running {len(instances)} instances × {runs} run(s) (concurrency={cfg['runner']['concurrency']})")
        for inst in instances[:5]:
            image = resolve_image(inst["docker_image"], cfg["docker"]["registry"])
            print(f"  {inst['instance_id']} [{inst.get('domain','')}] → {image}")
        if len(instances) > 5:
            print(f"  ... and {len(instances) - 5} more")
        if use_run_id:
            print(f"  Each instance will be evaluated {runs} times (run_1 .. run_{runs})")
        return

    scores_path = os.path.join(cfg["runner"]["output_dir"], cfg["dataset"]["config"], "scores.jsonl")
    os.makedirs(os.path.dirname(scores_path), exist_ok=True)

    all_results = []
    t0 = time.time()
    concurrency = cfg["runner"]["concurrency"]

    for run_id_val in range(1, runs + 1):
        cur_run_id = run_id_val if use_run_id else None
        if use_run_id:
            print(f"\n{'='*60}")
            print(f"  Run {run_id_val}/{runs}")
            print(f"{'='*60}")

        # Skip done (per run)
        if cfg["runner"]["skip_done"]:
            out_base = os.path.join(cfg["runner"]["output_dir"], cfg["dataset"]["config"])
            todo = []
            for inst in instances:
                if use_run_id:
                    result_path = os.path.join(out_base, inst["instance_id"],
                                               f"run_{run_id_val}", "result.json")
                else:
                    result_path = os.path.join(out_base, inst["instance_id"], "result.json")
                if os.path.exists(result_path):
                    with open(result_path) as f:
                        r = json.load(f)
                    if r.get("reward") is not None:
                        continue
                todo.append(inst)
            skipped = len(instances) - len(todo)
            if skipped:
                print(f"Skipping {skipped} already-completed instances")
            run_instances = todo
        else:
            run_instances = instances

        print(f"Running {len(run_instances)} instances (concurrency={concurrency})")

        if concurrency <= 1:
            for i, inst in enumerate(run_instances):
                result = eval_instance(inst, cfg, run_id=cur_run_id)
                all_results.append(result)
                with open(scores_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                r_str = f"{result['reward']:.3f}" if result["reward"] is not None else f"FAIL({result.get('phase','')}: {result.get('error','')})"
                run_tag = f" run={run_id_val}" if use_run_id else ""
                print(f"[{i+1}/{len(run_instances)}]{run_tag} {inst['instance_id']}: {r_str} ({result.get('duration_sec',0):.0f}s)", flush=True)
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as ex:
                futs = {ex.submit(eval_instance, inst, cfg, run_id=cur_run_id): inst for inst in run_instances}
                for i, fut in enumerate(as_completed(futs)):
                    result = fut.result()
                    all_results.append(result)
                    with open(scores_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    r_str = f"{result['reward']:.3f}" if result["reward"] is not None else f"FAIL({result.get('phase','')})"
                    run_tag = f" run={run_id_val}" if use_run_id else ""
                    print(f"[{i+1}/{len(run_instances)}]{run_tag} {result['instance_id']}: {r_str}", flush=True)

    _print_summary(all_results, cfg, time.time() - t0)


def cmd_status(args):
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)
    out_base = os.path.join(cfg["runner"]["output_dir"], cfg["dataset"]["config"])

    if not os.path.isdir(out_base):
        print(f"No results found at {out_base}")
        return

    scores_path = os.path.join(out_base, "scores.jsonl")
    if not os.path.exists(scores_path):
        print("No scores.jsonl found")
        return

    results = []
    with open(scores_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    _print_summary(results, cfg, None)


def cmd_report(args):
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)
    out_base = os.path.join(cfg["runner"]["output_dir"], cfg["dataset"]["config"])
    scores_path = os.path.join(out_base, "scores.jsonl")

    if not os.path.exists(scores_path):
        print(f"No scores.jsonl at {scores_path}")
        return

    results = []
    with open(scores_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    valid, avg, runs, run_scores, std = _aggregate_pass_at_n(results)
    failed = [r for r in results if r.get("reward") is None]

    summary = {
        "config": cfg["dataset"]["config"],
        "total": len(results),
        "completed": len(valid),
        "failed": len(failed),
        "avg_reward": round(avg, 4),
        "runs": runs,
        "std": round(std, 4),
        "run_scores": run_scores,
        "by_domain": {},
        "failure_breakdown": {},
    }

    # Per-domain
    domain_scores = {}
    for r in valid:
        d = r.get("domain", "unknown")
        domain_scores.setdefault(d, []).append(r["reward"])
    for d, scores in sorted(domain_scores.items()):
        summary["by_domain"][d] = {
            "count": len(scores),
            "avg_reward": round(sum(scores) / len(scores), 4),
            "min": round(min(scores), 4),
            "max": round(max(scores), 4),
        }

    # Failure breakdown
    for r in failed:
        phase = r.get("phase", "unknown")
        summary["failure_breakdown"][phase] = summary["failure_breakdown"].get(phase, 0) + 1

    summary_path = os.path.join(out_base, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved to {summary_path}")


def cmd_submit(args):
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)
    dataset_config = cfg["dataset"]["config"]
    out_base = os.path.join(cfg["runner"]["output_dir"], dataset_config)

    scores_path = os.path.join(out_base, "scores.jsonl")
    if not os.path.exists(scores_path):
        print(f"ERROR: no scores.jsonl at {scores_path}", file=sys.stderr)
        print("Run evaluation first: flyai-bench run", file=sys.stderr)
        sys.exit(1)

    results = []
    with open(scores_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    if not any(r.get("reward") is not None for r in results):
        print("ERROR: no valid scores found", file=sys.stderr)
        sys.exit(1)

    model_name = args.model or cfg["agent"].get("llm_model", "unknown")
    agent_type = args.agent_type or "mini-swe-agent"
    date_str = time.strftime("%Y%m%d")
    slug = f"{date_str}_{model_name}_{agent_type}".replace("/", "_").replace(" ", "_")
    submit_dir = os.path.join(args.submit_dir or "experiments/evaluation", dataset_config, slug)
    os.makedirs(submit_dir, exist_ok=True)

    import shutil
    shutil.copy2(scores_path, os.path.join(submit_dir, "scores.jsonl"))

    summary_path = os.path.join(out_base, "summary.json")
    if os.path.exists(summary_path):
        shutil.copy2(summary_path, os.path.join(submit_dir, "summary.json"))
    else:
        summary = _build_summary(results, dataset_config, model_name, agent_type)
        with open(os.path.join(submit_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    valid, avg_reward, runs, run_scores, std = _aggregate_pass_at_n(results)
    avg_reward = round(avg_reward, 4)
    failed = [r for r in results if r.get("reward") is None]
    eval_section = {
        "benchmark": "ecommerce_last_exam",
        "config": dataset_config,
        "total_instances": len(results),
        "completed": len(valid),
        "failed": len(failed),
        "avg_reward": avg_reward,
        "date": time.strftime("%Y-%m-%d"),
    }
    if runs > 1:
        eval_section["runs"] = runs
        eval_section["std"] = round(std, 4)
    metadata = {
        "model": {
            "name": model_name,
            "provider": args.provider or "",
            "api_params": {
                "temperature": 0.0,
                "max_tokens": 8192,
            },
        },
        "agent": {
            "type": agent_type,
            "version": "1.0",
            "max_iterations": 30,
            "tool_calling": "terminal",
        },
        "evaluation": eval_section,
        "notes": args.notes or "",
    }
    with open(os.path.join(submit_dir, "metadata.yaml"), "w", encoding="utf-8") as f:
        if yaml:
            yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        else:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Submission packaged at: {submit_dir}/")
    print(f"  metadata.yaml  — edit model/agent details before submitting")
    print(f"  scores.jsonl   — {len(results)} records ({len(valid)} valid, {len(failed)} failed)")
    print(f"  summary.json   — avg_reward={avg_reward:.4f}" + (f"  runs={runs}  std=±{std:.4f}" if runs > 1 else ""))
    print()
    print("Next steps:")
    print(f"  1. Review and edit {submit_dir}/metadata.yaml")
    print(f"  2. git add {submit_dir} && git commit")
    print(f"  3. Open a Pull Request to flyai-bench main branch")


def _build_summary(results, dataset_config, model_name, agent_type):
    valid, avg, runs, run_scores, std = _aggregate_pass_at_n(results)
    failed = [r for r in results if r.get("reward") is None]
    summary = {
        "config": dataset_config,
        "model": model_name,
        "agent": agent_type,
        "total": len(results),
        "completed": len(valid),
        "failed": len(failed),
        "avg_reward": round(avg, 4),
        "runs": runs,
        "std": round(std, 4),
        "run_scores": run_scores,
        "by_domain": {},
        "failure_breakdown": {},
    }
    domain_scores = {}
    for r in valid:
        d = r.get("domain", "unknown")
        domain_scores.setdefault(d, []).append(r["reward"])
    for d, scores in sorted(domain_scores.items()):
        summary["by_domain"][d] = {
            "count": len(scores),
            "avg_reward": round(sum(scores) / len(scores), 4),
            "min": round(min(scores), 4),
            "max": round(max(scores), 4),
        }
    for r in failed:
        phase = r.get("phase", "unknown")
        summary["failure_breakdown"][phase] = summary["failure_breakdown"].get(phase, 0) + 1
    return summary


def _aggregate_pass_at_n(results):
    """Aggregate grouped by instance_id, return (instance_means, runs, run_scores, std)."""
    valid = [r for r in results if r.get("reward") is not None]
    has_run_id = any(r.get("run_id") is not None for r in valid)

    if not has_run_id:
        avg = sum(r["reward"] for r in valid) / len(valid) if valid else 0
        return valid, avg, 1, [], 0.0

    from collections import defaultdict
    by_instance = defaultdict(list)
    by_run = defaultdict(list)
    for r in valid:
        by_instance[r["instance_id"]].append(r["reward"])
        by_run[r["run_id"]].append(r["reward"])

    instance_means = [sum(rs) / len(rs) for rs in by_instance.values()]
    avg = sum(instance_means) / len(instance_means) if instance_means else 0

    run_ids = sorted(by_run.keys())
    runs = len(run_ids)
    run_scores = [round(sum(by_run[rid]) / len(by_run[rid]), 4) for rid in run_ids]

    std = 0.0
    if runs > 1:
        mean_of_runs = sum(run_scores) / len(run_scores)
        variance = sum((s - mean_of_runs) ** 2 for s in run_scores) / (len(run_scores) - 1)
        std = variance ** 0.5

    return valid, avg, runs, run_scores, std


def _print_summary(results, cfg, elapsed):
    valid, avg, runs, run_scores, std = _aggregate_pass_at_n(results)
    failed_count = len(results) - len(valid)

    print(f"\n{'='*60}")
    print(f"  Config: {cfg['dataset']['config']}")
    print(f"  Completed: {len(valid)}/{len(results)}" + (f" (failed: {failed_count})" if failed_count else ""))
    print(f"  Avg Reward: {avg:.4f}")
    if runs > 1:
        print(f"  Runs: {runs}  Std: ±{std:.4f}")
        print(f"  Per-run scores: {', '.join(f'{s:.4f}' for s in run_scores)}")
    if elapsed:
        print(f"  Elapsed: {elapsed/60:.1f} min")

    domain_scores = {}
    for r in valid:
        d = r.get("domain", "unknown")
        domain_scores.setdefault(d, []).append(r["reward"])
    if domain_scores:
        print(f"  By domain:")
        for d, scores in sorted(domain_scores.items()):
            print(f"    {d}: {sum(scores)/len(scores):.4f} (n={len(scores)})")
    print(f"{'='*60}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="flyai-bench",
        description="E-Commerce Last Exam Benchmark Evaluation CLI",
    )
    default_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_config.yaml")
    parser.add_argument("--config", default=default_config,
                        help="Config file path (default: built-in eval_config.yaml)")

    sub = parser.add_subparsers(dest="command")

    # run
    p_run = sub.add_parser("run", help="Run evaluation")
    p_run.add_argument("--dataset-config", choices=["travel", "e_commerce"])
    p_run.add_argument("--limit", type=int)
    p_run.add_argument("--concurrency", type=int)
    p_run.add_argument("--output-dir")
    p_run.add_argument("--agent",
                        help="Named custom Agent registered in the config file")
    p_run.add_argument("--registry")
    p_run.add_argument("--runs", type=int, default=1,
                        help="Number of independent runs (pass@N, default: 1)")
    p_run.add_argument("--dry-run", action="store_true")

    # status
    p_status = sub.add_parser("status", help="Check evaluation progress")
    p_status.add_argument("--dataset-config", choices=["travel", "e_commerce"])
    p_status.add_argument("--output-dir")

    # report
    p_report = sub.add_parser("report", help="Generate an evaluation report")
    p_report.add_argument("--dataset-config", choices=["travel", "e_commerce"])
    p_report.add_argument("--output-dir")

    # submit
    p_submit = sub.add_parser("submit", help="Package evaluation results for leaderboard submission")
    p_submit.add_argument("--dataset-config", choices=["travel", "e_commerce"])
    p_submit.add_argument("--output-dir")
    p_submit.add_argument("--submit-dir", default="experiments/evaluation",
                          help="Output directory for the submission package (default: experiments/evaluation/)")
    p_submit.add_argument("--model", help="Model name (e.g. deepseek-v4-flash)")
    p_submit.add_argument("--provider", help="Model provider (e.g. deepseek)")
    p_submit.add_argument("--agent-type", default="mini-swe-agent",
                          help="Agent type (default: mini-swe-agent)")
    p_submit.add_argument("--notes", help="Submission notes")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "run":
        cmd_run(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "report":
        cmd_report(args)
    elif args.command == "submit":
        cmd_submit(args)


if __name__ == "__main__":
    main()
