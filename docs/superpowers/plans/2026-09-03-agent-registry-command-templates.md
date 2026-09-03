# Agent Registry and Command Templates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add named custom Agent registrations with safe command placeholders and one-flag selection while preserving the built-in container Agent.

**Architecture:** Keep the existing context manifest as the protocol boundary. Add pure config-resolution and command-rendering helpers to `flyai_bench.cli`, then route registered Agents through the existing external evaluation lifecycle using `shell=False` and a config-relative working directory.

**Tech Stack:** Python 3.9+, argparse, PyYAML, subprocess, pytest

## Global Constraints

- Custom Agents are selected only through `agent.name` or `--agent`; legacy mode and free-form command fields are removed.
- Registered Agent commands run with `shell=False`.
- Secrets never enter placeholders or `agent_context.json`.
- `answer.json` and `answer.md` are always required; extra outputs stay under `artifacts/`.
- Root compatibility copies must match package files byte-for-byte.

---

### Task 1: Config Registry and Selection

**Files:**
- Modify: `tests/test_external_agent_protocol.py`
- Modify: `src/flyai_bench/cli.py`

**Interfaces:**
- Produces: `resolve_registered_agent(cfg: dict, selected_name: str | None) -> dict | None`
- Produces: `load_config(path)` preserving `agents` and `_config_dir`

- [x] Write tests proving top-level `agents` survive loading, config-relative metadata is retained, CLI selection overrides `agent.name`, unknown names fail, and legacy configs remain unchanged.
- [x] Run focused tests and confirm they fail because registry resolution is absent.
- [x] Implement minimal validation and runtime merge for `protocol`, `timeout_sec`, and `output_paths`.
- [x] Run focused tests and confirm they pass.

### Task 2: Safe Command Templates

**Files:**
- Modify: `tests/test_external_agent_protocol.py`
- Modify: `src/flyai_bench/cli.py`

**Interfaces:**
- Produces: `agent_command_placeholders(...) -> dict[str, str]`
- Produces: `render_registered_agent_command(command, placeholders) -> list[str]`
- Produces: `resolve_registered_agent_cwd(registration, config_dir) -> str`

- [x] Write tests for argv-list rendering, string parsing, spaces in paths, every supported placeholder, unknown placeholders, empty commands, and config-relative working directories.
- [x] Run focused tests and confirm expected missing-helper failures.
- [x] Implement rendering with `str.format_map`, explicit allowed names, and `shlex.split` for string commands.
- [x] Run focused tests and confirm they pass.

### Task 3: Explicit Environment Passing

**Files:**
- Modify: `tests/test_external_agent_protocol.py`
- Modify: `src/flyai_bench/cli.py`

**Interfaces:**
- Produces: `build_registered_agent_env(cfg, registration, base_env=None) -> dict[str, str]`

- [x] Write tests proving execution essentials, configured `LLM_*`, and allowlisted variables pass while unrelated variables and invalid names do not.
- [x] Run focused tests and confirm expected failures.
- [x] Implement the minimal environment builder without persisting secrets.
- [x] Run focused tests and confirm they pass.

### Task 4: Runtime and CLI Integration

**Files:**
- Modify: `tests/test_external_agent_protocol.py`
- Modify: `src/flyai_bench/cli.py`
- Modify: `run_eval.py`

**Interfaces:**
- Consumes: registry, placeholder, working-directory, and environment helpers from Tasks 1-3
- Produces: `flyai-bench --config FILE run --agent NAME`

- [x] Write a subprocess-capture test proving registered Agents launch with rendered argv, `shell=False`, resolved `cwd`, and registered environment.
- [x] Add `--agent`, resolve registrations in `apply_overrides`, and branch registered launches inside the existing external flow.
- [x] Remove the legacy external-command path and synchronize `run_eval.py`.
- [x] Run all protocol tests.

### Task 5: Example and Documentation

**Files:**
- Modify: `examples/external_agent.py`
- Modify: `eval_config.yaml`
- Modify: `src/flyai_bench/eval_config.yaml`
- Modify: `README.md`

**Interfaces:**
- Produces: example invocation `python3 examples/external_agent.py --context "{context}"`

- [x] Add tests for the example context argument with environment fallback.
- [x] Update the example to parse `--context` and retain `FLYAI_BENCH_CONTEXT` fallback.
- [x] Add a copyable Agent registration to both config templates and make it the primary README integration path.
- [x] Run documentation command examples through parser/dry-run checks.

### Task 6: Verification

**Files:**
- Verify: `tests/test_external_agent_protocol.py`
- Verify: package and root compatibility files

**Interfaces:**
- Consumes: all previous tasks
- Produces: a regression-tested package ready for a later explicit release request

- [x] Run `pytest -q` and require all tests to pass.
- [x] Run `python -m compileall src examples run_eval.py`.
- [x] Verify `run_eval.py`, `tool_server.py`, and `sandbox_setup.sh` match package copies where required.
- [x] Build wheel and sdist, run `twine check`, and inspect the wheel contents.
- [x] Run registered-Agent smoke evaluations for one `travel` and one `e_commerce` task.
