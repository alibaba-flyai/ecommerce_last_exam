import json
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from flyai_bench import cli
from flyai_bench import tool_server


def _load_external_agent_example():
    path = Path(__file__).parents[1] / "examples" / "external_agent.py"
    spec = importlib.util.spec_from_file_location("external_agent_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_config_preserves_agent_registry_and_config_directory(tmp_path):
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(
        """
agent:
  name: my-agent
agents:
  my-agent:
    protocol: manifest-v1
    command: [python3, ./agent.py, --context, "{context}"]
""".strip(),
        encoding="utf-8",
    )

    cfg = cli.load_config(str(config_path))

    assert cfg["agent"]["name"] == "my-agent"
    assert cfg["agents"]["my-agent"]["command"][1] == "./agent.py"
    assert cfg["_config_dir"] == str(tmp_path.resolve())


def test_registered_agent_selection_activates_registration():
    cfg = {
        "agent": {"name": "configured", "timeout_sec": 100},
        "agents": {
            "configured": {"command": ["configured-agent"]},
            "selected": {
                "protocol": "manifest-v1",
                "command": ["selected-agent", "{context}"],
                "timeout_sec": 240,
                "output_paths": ["artifacts/trajectory.jsonl"],
            },
        },
        "_config_dir": "/tmp/config",
    }

    registration = cli.resolve_registered_agent(cfg, "selected")

    assert registration["name"] == "selected"
    assert registration["protocol"] == "manifest-v1"
    assert cfg["agent"]["name"] == "selected"
    assert cfg["agent"]["timeout_sec"] == 240
    assert cfg["agent"]["output_paths"] == ["artifacts/trajectory.jsonl"]
    assert cfg["agent"]["registration"] is registration


def test_registered_agent_selection_uses_agent_name_from_config():
    cfg = {
        "agent": {"name": "my-agent"},
        "agents": {"my-agent": {"command": ["python3", "agent.py"]}},
    }

    registration = cli.resolve_registered_agent(cfg)

    assert registration["name"] == "my-agent"
    assert cfg["agent"]["registration"] is registration


def test_registered_agent_selection_rejects_unknown_name():
    cfg = {"agent": {}, "agents": {"known": {"command": ["agent"]}}}

    with pytest.raises(ValueError, match="unknown Agent 'missing'.*known"):
        cli.resolve_registered_agent(cfg, "missing")


def test_registered_agent_selection_rejects_unsupported_protocol():
    cfg = {
        "agent": {},
        "agents": {"my-agent": {"protocol": "acp", "command": ["agent"]}},
    }

    with pytest.raises(ValueError, match="unsupported protocol"):
        cli.resolve_registered_agent(cfg, "my-agent")


@pytest.mark.parametrize("timeout_sec", [0, -1, "1800", True])
def test_registered_agent_selection_rejects_invalid_timeout(timeout_sec):
    cfg = {
        "agent": {},
        "agents": {
            "my-agent": {"command": ["agent"], "timeout_sec": timeout_sec},
        },
    }

    with pytest.raises(ValueError, match="timeout_sec must be a positive number"):
        cli.resolve_registered_agent(cfg, "my-agent")


def test_no_registered_agent_keeps_internal_agent_config_unchanged():
    cfg = {"agent": {"cmd": "python3 /app/mini_swe_agent.py"}}

    assert cli.resolve_registered_agent(cfg) is None
    assert cfg == {"agent": {"cmd": "python3 /app/mini_swe_agent.py"}}


def test_default_config_omits_legacy_agent_mode_and_external_command():
    assert "mode" not in cli.DEFAULT_CONFIG["agent"]
    assert "external_cmd" not in cli.DEFAULT_CONFIG["agent"]


@pytest.mark.parametrize("field", ["mode", "external_cmd"])
def test_load_config_rejects_removed_legacy_agent_fields(tmp_path, field):
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(
        f"agent:\n  {field}: external\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=rf"unsupported agent field.*{field}"):
        cli.load_config(str(config_path))


def test_registered_command_renders_argv_without_shell_splitting_paths():
    placeholders = {
        "context": "/tmp/output with spaces/.prompt/agent_context.json",
        "instance_id": "task-1",
    }

    command = cli.render_registered_agent_command(
        ["python3", "./agent.py", "--context", "{context}", "--task={instance_id}"],
        placeholders,
    )

    assert command == [
        "python3",
        "./agent.py",
        "--context",
        "/tmp/output with spaces/.prompt/agent_context.json",
        "--task=task-1",
    ]


def test_registered_command_accepts_shell_style_string_without_using_shell():
    command = cli.render_registered_agent_command(
        'python3 "./my agent.py" --context {context}',
        {"context": "/tmp/context.json"},
    )

    assert command == ["python3", "./my agent.py", "--context", "/tmp/context.json"]


def test_registered_command_rejects_unknown_placeholder():
    with pytest.raises(ValueError, match="unknown Agent command placeholder.*secret"):
        cli.render_registered_agent_command(
            ["agent", "--token", "{secret}"],
            {"context": "/tmp/context.json"},
        )


@pytest.mark.parametrize("command", [None, "", [], ["agent", 3]])
def test_registered_command_rejects_invalid_values(command):
    with pytest.raises(ValueError, match="Agent command"):
        cli.render_registered_agent_command(command, {"context": "/tmp/context.json"})


def test_agent_command_placeholders_expose_manifest_and_materialized_files(tmp_path):
    prompt_dir = tmp_path / "output" / ".prompt"
    output_dir = tmp_path / "output"
    placeholders = cli.agent_command_placeholders(
        instance_id="task-1",
        tool_server_url="http://localhost:49152",
        output_dir=str(output_dir),
        prompt_dir=str(prompt_dir),
        agent_context_path=str(prompt_dir / "agent_context.json"),
        config_dir=str(tmp_path),
    )

    assert placeholders == {
        "context": str(prompt_dir / "agent_context.json"),
        "output_dir": str(output_dir),
        "instance_id": "task-1",
        "tool_server_url": "http://localhost:49152",
        "system_prompt_path": str(prompt_dir / "system.md"),
        "instruction_path": str(prompt_dir / "instruction.md"),
        "tool_defs_path": str(prompt_dir / "tool_defs.json"),
        "task_context_path": str(prompt_dir / "context.json"),
        "config_dir": str(tmp_path),
    }


def test_registered_agent_cwd_is_relative_to_config_file(tmp_path):
    agent_dir = tmp_path / "agents" / "sample"
    agent_dir.mkdir(parents=True)

    assert cli.resolve_registered_agent_cwd(
        {"cwd": "agents/sample"}, str(tmp_path)
    ) == str(agent_dir.resolve())
    assert cli.resolve_registered_agent_cwd({}, str(tmp_path)) == str(tmp_path.resolve())


def test_registered_agent_cwd_must_be_an_existing_directory(tmp_path):
    with pytest.raises(ValueError, match="working directory does not exist"):
        cli.resolve_registered_agent_cwd({"cwd": "missing"}, str(tmp_path))


def test_registered_agent_environment_uses_explicit_allowlist(monkeypatch):
    base_env = {
        "PATH": "/usr/bin",
        "HOME": "/home/user",
        "OPENAI_API_KEY": "openai-secret",
        "UNRELATED_SECRET": "must-not-pass",
        "LLM_API_KEY": "shell-llm-secret",
    }
    cfg = {
        "agent": {
            "llm_base_url": "https://llm.example/v1",
            "llm_api_key": "",
            "llm_model": "example-model",
        }
    }
    registration = {"pass_env": ["OPENAI_API_KEY"]}

    env = cli.build_registered_agent_env(cfg, registration, base_env=base_env)

    assert env["PATH"] == "/usr/bin"
    assert env["HOME"] == "/home/user"
    assert env["OPENAI_API_KEY"] == "openai-secret"
    assert env["LLM_BASE_URL"] == "https://llm.example/v1"
    assert env["LLM_API_KEY"] == "shell-llm-secret"
    assert env["LLM_MODEL"] == "example-model"
    assert "UNRELATED_SECRET" not in env


@pytest.mark.parametrize("pass_env", ["OPENAI_API_KEY", ["BAD-NAME"], [3]])
def test_registered_agent_environment_rejects_invalid_pass_env(pass_env):
    with pytest.raises(ValueError, match="pass_env"):
        cli.build_registered_agent_env(
            {"agent": {}}, {"pass_env": pass_env}, base_env={"PATH": os.defpath}
        )


def test_registered_agent_launch_uses_argv_cwd_environment_and_no_shell(
        monkeypatch, tmp_path):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    cfg = {
        "agent": {
            "llm_base_url": "https://llm.example/v1",
            "llm_api_key": "secret",
            "llm_model": "model",
            "timeout_sec": 123,
        },
        "_config_dir": str(tmp_path),
    }
    registration = {
        "command": ["python3", "agent.py", "--context", "{context}"],
        "cwd": ".",
        "pass_env": [],
    }

    result = cli.run_registered_agent(
        cfg,
        registration,
        {"context": "/tmp/agent_context.json"},
        base_env={"PATH": "/usr/bin"},
    )

    assert result.returncode == 0
    assert captured["command"] == [
        "python3", "agent.py", "--context", "/tmp/agent_context.json"]
    assert captured["shell"] is False
    assert captured["cwd"] == str(tmp_path.resolve())
    assert captured["env"]["LLM_API_KEY"] == "secret"
    assert captured["timeout"] == 123
    assert captured["capture_output"] is True
    assert captured["text"] is True


def test_cli_accepts_registered_agent_name(monkeypatch):
    captured = {}

    def fake_cmd_run(args):
        captured["agent"] = args.agent

    monkeypatch.setattr(cli, "cmd_run", fake_cmd_run)
    monkeypatch.setattr(cli.sys, "argv", ["flyai-bench", "run", "--agent", "my-agent"])

    cli.main()

    assert captured["agent"] == "my-agent"


@pytest.mark.parametrize(
    ("flag", "value"),
    [("--agent-mode", "external"), ("--agent-cmd", "python3 agent.py")],
)
def test_cli_rejects_removed_legacy_agent_flags(monkeypatch, flag, value):
    monkeypatch.setattr(cli, "cmd_run", lambda args: None)
    monkeypatch.setattr(
        cli.sys, "argv", ["flyai-bench", "run", flag, value, "--dry-run"])

    with pytest.raises(SystemExit) as exc_info:
        cli.main()

    assert exc_info.value.code == 2


def test_external_launch_routes_registered_agent_through_template_runner(
        monkeypatch, tmp_path):
    captured = {}

    def fake_registered(cfg, registration, placeholders, base_env=None):
        captured["registration"] = registration
        captured["placeholders"] = placeholders
        return SimpleNamespace(returncode=0, stdout="registered", stderr="")

    monkeypatch.setattr(cli, "run_registered_agent", fake_registered)
    cfg = {
        "agent": {
            "registration": {"name": "my-agent", "command": ["agent", "{context}"]},
            "timeout_sec": 300,
        },
        "_config_dir": str(tmp_path),
    }
    prompt_dir = tmp_path / "result" / ".prompt"

    result = cli.launch_external_agent(
        cfg,
        instance_id="task-1",
        tool_server_url="http://localhost:49152",
        output_dir=str(tmp_path / "result"),
        prompt_dir=str(prompt_dir),
        output_paths=["answer.json", "answer.md"],
        agent_context_path=str(prompt_dir / "agent_context.json"),
    )

    assert result.stdout == "registered"
    assert captured["registration"]["name"] == "my-agent"
    assert captured["placeholders"]["context"] == str(
        (prompt_dir / "agent_context.json").resolve())
    assert captured["placeholders"]["config_dir"] == str(tmp_path.resolve())


def test_external_launch_requires_registered_agent(tmp_path):
    cfg = {
        "agent": {
            "timeout_sec": 300,
        },
    }
    prompt_dir = tmp_path / ".prompt"

    with pytest.raises(ValueError, match="registered Agent is not selected"):
        cli.launch_external_agent(
            cfg,
            instance_id="task-1",
            tool_server_url="http://localhost:49152",
            output_dir=str(tmp_path),
            prompt_dir=str(prompt_dir),
            output_paths=["answer.json", "answer.md"],
            agent_context_path=str(prompt_dir / "agent_context.json"),
        )


def test_external_agent_example_reads_context_argument(tmp_path):
    example = _load_external_agent_example()
    context_path = tmp_path / "agent_context.json"
    context_path.write_text('{"protocol_version": "1.0"}', encoding="utf-8")

    assert example.read_context(str(context_path)) == {"protocol_version": "1.0"}


def test_external_agent_example_accepts_explicit_turn_budget():
    example = _load_external_agent_example()

    args = example.parse_args([
        "--context", "/tmp/context.json", "--max-turns", "12"])

    assert args.context == "/tmp/context.json"
    assert args.max_turns == 12


def test_external_agent_example_defaults_to_30_turns_without_env_override(
        monkeypatch):
    monkeypatch.setenv("MAX_TURNS", "7")
    example = _load_external_agent_example()

    args = example.parse_args(["--context", "/tmp/context.json"])

    assert args.max_turns == 30


def test_external_agent_example_keeps_environment_fallback(monkeypatch, tmp_path):
    example = _load_external_agent_example()
    context_path = tmp_path / "agent_context.json"
    context_path.write_text('{"protocol_version": "1.0"}', encoding="utf-8")
    monkeypatch.setenv("FLYAI_BENCH_CONTEXT", str(context_path))

    assert example.read_context() == {"protocol_version": "1.0"}


def test_external_agent_example_records_declared_trajectory(tmp_path):
    example = _load_external_agent_example()
    context = {
        "output": {
            "directory": str(tmp_path),
            "paths": ["answer.json", "answer.md", "artifacts/trajectory.jsonl"],
        }
    }

    trajectory_path = example.prepare_trajectory(context)
    example.append_trajectory_event(
        trajectory_path,
        {"turn": 1, "event": "assistant", "message": {"content": "hello"}},
    )

    assert trajectory_path == tmp_path / "artifacts" / "trajectory.jsonl"
    assert json.loads(trajectory_path.read_text(encoding="utf-8")) == {
        "turn": 1,
        "event": "assistant",
        "message": {"content": "hello"},
    }


def test_external_agent_example_skips_undeclared_trajectory(tmp_path):
    example = _load_external_agent_example()
    context = {
        "output": {
            "directory": str(tmp_path),
            "paths": ["answer.json", "answer.md"],
        }
    }

    assert example.prepare_trajectory(context) is None
    example.append_trajectory_event(None, {"event": "ignored"})
    assert not (tmp_path / "artifacts" / "trajectory.jsonl").exists()


def test_external_agent_example_forces_final_answer_without_tools():
    example = _load_external_agent_example()
    captured = {}
    message = SimpleNamespace(
        content='{"feasible": false}',
        model_dump=lambda **kwargs: {"content": '{"feasible": false}'},
    )

    def fake_create(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))
    messages = [{"role": "user", "content": "task"}]

    result = example.request_final_answer(
        client,
        model="example-model",
        messages=messages,
        trajectory_path=None,
        turn=31,
    )

    assert result == '{"feasible": false}'
    assert captured["tool_choice"] == "none"
    assert captured["messages"][-1]["role"] == "user"
    assert "stop calling tools" in captured["messages"][-1]["content"].lower()


def test_external_agent_example_splits_model_file_payload(tmp_path):
    example = _load_external_agent_example()
    example.write_answers(
        str(tmp_path),
        {
            "answer_json": {"feasible": False},
            "answer_md": "# Recommendation\n\nThe requested schedule is not feasible.",
        },
        "unused raw response",
    )

    assert json.loads((tmp_path / "answer.json").read_text(encoding="utf-8")) == {
        "feasible": False,
    }
    assert (tmp_path / "answer.md").read_text(encoding="utf-8") == (
        "# Recommendation\n\nThe requested schedule is not feasible.\n")


def test_eval_instance_persists_result_when_pull_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "docker_pull", lambda image: (False, "not found"))
    monkeypatch.setattr(cli, "docker_rm", lambda name: None)
    cfg = {
        "dataset": {"config": "travel"},
        "docker": {"registry": "", "platform": "linux/amd64"},
        "agent": {"mode": "internal"},
        "runner": {"output_dir": str(tmp_path)},
    }

    result = cli.eval_instance(
        {
            "instance_id": "task-1",
            "domain": "travel",
            "docker_image": "example/missing:latest",
        },
        cfg,
    )

    result_path = tmp_path / "travel" / "task-1" / "result.json"
    assert result["phase"] == "pull"
    assert json.loads(result_path.read_text(encoding="utf-8"))["phase"] == "pull"


def test_builtin_config_contains_copyable_registered_agent_example():
    config_path = Path(cli.__file__).with_name("eval_config.yaml")
    cfg = cli.load_config(str(config_path))

    registration = cfg["agents"]["example-openai-agent"]
    assert registration["protocol"] == "manifest-v1"
    assert registration["command"][:3] == [
        "python3", "-m", "flyai_bench.external_agent"]
    assert registration["command"][-3:] == ["{context}", "--max-turns", "30"]
    assert "name" not in cfg["agent"]


def test_external_agent_example_is_packaged_as_a_module():
    assert importlib.util.find_spec("flyai_bench.external_agent") is not None


def test_package_declares_huggingface_dataset_reader_dependency():
    pyproject_path = Path(__file__).parents[1] / "pyproject.toml"
    pyproject = pyproject_path.read_text(encoding="utf-8")

    assert '"datasets>=' in pyproject


def test_tool_command_preserves_schema_property_names_and_json_values():
    build = getattr(tool_server, "build_tool_command", None)
    assert build is not None

    command = build(
        "/app/.tools_real/search",
        {
            "tool_call_list": [{"cityName": "Hangzhou"}],
            "cityName": "Hangzhou",
            "isAbroad": False,
        },
    )

    assert command == [
        "/app/.tools_real/search",
        "--tool_call_list",
        '[{"cityName": "Hangzhou"}]',
        "--cityName",
        "Hangzhou",
        "--isAbroad",
        "false",
    ]


@pytest.mark.parametrize("tool_name", ["../answer.json", "/bin/sh", ".hidden"])
def test_tool_name_validation_rejects_path_traversal(tool_name):
    validate = getattr(tool_server, "validate_tool_name", None)
    assert validate is not None

    with pytest.raises(ValueError):
        validate(tool_name)


def test_tool_server_does_not_own_the_verifier_audit_log():
    assert not hasattr(tool_server, "TOOL_PROXY_LOG")
    assert not hasattr(tool_server, "log_proxy_tool_call")


def test_external_output_paths_include_required_files_and_extras():
    normalize = getattr(cli, "external_output_paths", None)
    assert normalize is not None

    assert normalize({}) == ["answer.json", "answer.md"]
    assert normalize({"output_paths": ["artifacts/evidence/sources.json", "answer.json"]}) == [
        "answer.json",
        "answer.md",
        "artifacts/evidence/sources.json",
    ]
    assert normalize({"output_path": "artifacts/trace/events.jsonl"}) == [
        "answer.json",
        "answer.md",
        "artifacts/trace/events.jsonl",
    ]
    assert normalize({
        "output_paths": [],
        "output_path": "artifacts/legacy.json",
    }) == [
        "answer.json",
        "answer.md",
        "artifacts/legacy.json",
    ]


@pytest.mark.parametrize("path", [
    "", "/tmp/result.json", "../secret", "a/../../secret",
    "reward.txt", "tests/test.sh", "evidence.json",
])
def test_external_output_paths_reject_unsafe_paths(path):
    normalize = getattr(cli, "external_output_paths", None)
    assert normalize is not None

    with pytest.raises(ValueError):
        normalize({"output_paths": [path]})


def test_missing_external_outputs_reports_every_missing_file(tmp_path):
    missing = getattr(cli, "missing_external_outputs", None)
    assert missing is not None

    (tmp_path / "answer.json").write_text("{}", encoding="utf-8")
    assert missing(tmp_path, ["answer.json", "answer.md", "evidence.json"]) == [
        "answer.md",
        "evidence.json",
    ]


def test_clear_external_outputs_removes_stale_declared_files_only(tmp_path):
    clear = getattr(cli, "clear_external_outputs", None)
    assert clear is not None

    (tmp_path / "answer.json").write_text("stale", encoding="utf-8")
    (tmp_path / "keep.log").write_text("keep", encoding="utf-8")
    nested = tmp_path / "artifacts" / "evidence" / "sources.json"
    nested.parent.mkdir(parents=True)
    nested.write_text("stale", encoding="utf-8")

    clear(tmp_path, ["answer.json", "answer.md", "artifacts/evidence/sources.json"])

    assert not (tmp_path / "answer.json").exists()
    assert not nested.exists()
    assert (tmp_path / "keep.log").read_text(encoding="utf-8") == "keep"


def test_clear_external_outputs_does_not_follow_parent_symlinks(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    protected = outside / "result.json"
    protected.write_text("keep", encoding="utf-8")
    (output_dir / "artifacts").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError):
        cli.clear_external_outputs(output_dir, ["artifacts/result.json"])

    assert protected.read_text(encoding="utf-8") == "keep"


def test_copy_external_outputs_copies_each_declared_file(monkeypatch, tmp_path):
    (tmp_path / "answer.json").write_text("{}", encoding="utf-8")
    (tmp_path / "answer.md").write_text("summary", encoding="utf-8")
    evidence = tmp_path / "artifacts" / "evidence" / "sources.json"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("[]", encoding="utf-8")

    exec_calls = []
    copy_calls = []

    def fake_exec(container_name, command, timeout):
        exec_calls.append((container_name, command, timeout))
        return 0, "", ""

    def fake_run(command, **kwargs):
        copy_calls.append(command)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(cli, "docker_exec", fake_exec)
    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    ok, error = cli.copy_external_outputs(
        "eval-task",
        str(tmp_path),
        ["answer.json", "answer.md", "artifacts/evidence/sources.json"],
    )

    assert ok is True
    assert error == ""
    assert [call[-1] for call in copy_calls] == [
        "eval-task:/app/answer.json",
        "eval-task:/app/answer.md",
        "eval-task:/app/artifacts/evidence/sources.json",
    ]
    assert exec_calls == [("eval-task", "mkdir -p -- /app/artifacts/evidence", 10)]


def test_legacy_external_environment_builder_is_removed():
    assert not hasattr(cli, "build_external_agent_env")


def test_external_tool_port_is_published_on_loopback_only():
    publish_spec = getattr(cli, "external_tool_port_publish_spec", None)
    assert publish_spec is not None
    assert publish_spec() == "127.0.0.1::9999"


def test_sandbox_wraps_only_regular_tool_files():
    sandbox = Path(cli.__file__).with_name("sandbox_setup.sh").read_text(encoding="utf-8")

    assert '[ -f "$tool_path" ] || continue' in sandbox
    assert 'key = args[i][2:]' in sandbox
    assert '.replace("-", "_")' not in sandbox


def test_agent_context_manifest_contains_protocol_without_persisting_api_key(tmp_path):
    write_context = getattr(cli, "write_external_agent_context", None)
    assert write_context is not None

    prompt_dir = tmp_path / ".prompt"
    prompt_dir.mkdir()
    (prompt_dir / "system.md").write_text("System rules", encoding="utf-8")
    (prompt_dir / "instruction.md").write_text("User task", encoding="utf-8")
    (prompt_dir / "tool_defs.json").write_text(
        '[{"name": "search", "parameters": {}}]', encoding="utf-8")
    (prompt_dir / "context.json").write_text('{"region": "APAC"}', encoding="utf-8")
    cfg = {
        "agent": {
            "llm_base_url": "https://llm.example/v1",
            "llm_api_key": "must-not-be-written",
            "llm_model": "example-model",
        }
    }

    context_path = write_context(
        cfg,
        instance_id="task-1",
        tool_server_url="http://localhost:49152",
        output_dir=str(tmp_path),
        prompt_dir=str(prompt_dir),
        output_paths=["answer.json", "answer.md", "artifacts/evidence/sources.json"],
    )
    raw = Path(context_path).read_text(encoding="utf-8")
    context = json.loads(raw)

    assert context["protocol_version"] == "1.0"
    assert context["instance_id"] == "task-1"
    assert context["prompts"] == {"system": "System rules", "user": "User task"}
    assert context["tools"]["url"] == "http://localhost:49152"
    assert context["tools"]["definitions"][0]["name"] == "search"
    assert context["task_context"] == {"region": "APAC"}
    assert context["output"]["paths"] == [
        "answer.json", "answer.md", "artifacts/evidence/sources.json"]
    assert context["llm"] == {
        "base_url": "https://llm.example/v1",
        "model": "example-model",
        "api_key_env": "LLM_API_KEY",
    }
    assert "must-not-be-written" not in raw


@pytest.mark.parametrize("contents", [None, "[]"])
def test_agent_context_requires_non_empty_tool_definitions(tmp_path, contents):
    prompt_dir = tmp_path / ".prompt"
    prompt_dir.mkdir()
    (prompt_dir / "instruction.md").write_text("User task", encoding="utf-8")
    if contents is not None:
        (prompt_dir / "tool_defs.json").write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match="tool_defs.json"):
        cli.write_external_agent_context(
            {"agent": {}},
            instance_id="task-1",
            tool_server_url="http://localhost:49152",
            output_dir=str(tmp_path),
            prompt_dir=str(prompt_dir),
            output_paths=["answer.json", "answer.md"],
        )
