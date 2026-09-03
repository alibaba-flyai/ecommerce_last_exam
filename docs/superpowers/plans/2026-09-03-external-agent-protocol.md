# External Agent Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `flyai-bench` external Agent evaluation reliable, multi-file capable, documented, and ready for a PyPI release.

**Architecture:** The CLI launches the benchmark container and its HTTP Tool Server, then starts the user Agent on the host with prompt and output locations injected as environment variables. The Agent calls container tools over localhost HTTP and writes declared relative output paths under `OUTPUT_DIR`; the CLI validates and copies every declared file into `/app` before running the verifier.

**Tech Stack:** Python 3.10+, argparse, Docker CLI, stdlib HTTP server, pytest, YAML.

## Global Constraints

- Preserve tool parameter names exactly as defined by `tool_defs.json`.
- Require `answer.json` and `answer.md` for every external Agent run.
- Permit additional output files only as safe relative paths below `OUTPUT_DIR/artifacts/`.
- Keep `src/flyai_bench/` and root standalone compatibility files synchronized.
- Do not expose database or verifier files to the external Agent.

---

### Task 1: Tool Server Argument Rendering

**Files:**
- Modify: `src/flyai_bench/tool_server.py`
- Modify: `tool_server.py`
- Test: `tests/test_external_agent_protocol.py`

**Interfaces:**
- Consumes: tool name and JSON object arguments from `POST /call/<tool_name>`.
- Produces: `build_tool_command(tool_path: str, arguments: dict) -> list[str]`.

- [ ] Write tests for exact snake_case/camelCase flags, JSON values, booleans, and unsafe tool names.
- [ ] Run the focused tests and confirm they fail on the current kebab-case conversion.
- [ ] Add command rendering and tool-name validation.
- [ ] Run the focused tests and confirm they pass.

### Task 2: Multi-File External Outputs

**Files:**
- Modify: `src/flyai_bench/cli.py`
- Modify: `src/flyai_bench/eval_config.yaml`
- Modify: `run_eval.py`
- Modify: `eval_config.yaml`
- Test: `tests/test_external_agent_protocol.py`

**Interfaces:**
- Consumes: `agent.output_paths` as a string or list of safe relative paths.
- Produces: `external_output_paths(agent_cfg) -> list[str]` containing required answers and declared extras.
- Produces: `OUTPUT_PATHS_JSON` plus prompt paths and LLM settings in the external process environment.

- [ ] Write tests for defaults, extra paths, legacy singular input, traversal rejection, missing-file detection, and injected environment variables.
- [ ] Run tests and confirm they fail because the protocol helpers do not exist.
- [ ] Implement normalization, validation, environment construction, and copying every output to `/app`.
- [ ] Copy the tool-call log back to the host for consistent audit artifacts.
- [ ] Run focused and full tests.

### Task 3: Example, Documentation, and Release Metadata

**Files:**
- Modify: `examples/external_agent.py`
- Modify: `README.md`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: the environment-variable protocol from Task 2.
- Produces: a runnable external Agent example and public integration instructions for version `0.1.2`.

- [ ] Update the example to read both prompts, call tools with schema names, and emit `answer.json` plus `answer.md`.
- [ ] Document source installation, configuration, execution, reports, submissions, concurrency, and output rules.
- [ ] Bump the package version to `0.1.2`.
- [ ] Run unit tests, syntax checks, package build, wheel-content checks, and a one-task external-mode integration test.
- [ ] Review the final diff and commit the verified implementation.
