# Agent Registry and Command Templates Design

## Goal

Make custom Agent integration a single-config workflow. Users register a named
host-side Agent, select it by name, and receive the benchmark context through a
command placeholder instead of wiring benchmark environment variables by hand.

## Configuration

The existing `agent` mapping remains the runtime configuration and gains an
optional `name`. A new top-level `agents` mapping contains named registrations:

```yaml
agent:
  name: my-agent
  llm_base_url: https://api.example.com/v1
  llm_api_key: ""
  llm_model: example-model

agents:
  my-agent:
    protocol: manifest-v1
    command:
      - python3
      - ./my_agent.py
      - --context
      - "{context}"
      - --max-turns
      - "30"
    cwd: .
    timeout_sec: 1800
    output_paths:
      - artifacts/trajectory.jsonl
    pass_env:
      - OPENAI_API_KEY
```

`agent.name` may be overridden with `flyai-bench ... run --agent my-agent`.
Selecting a registered Agent automatically runs it on the host. When no Agent is
selected, the built-in container Agent runs. Legacy mode fields and free-form
command overrides are removed so Agent selection has one explicit interface.

## Command Model

`command` should be a YAML list so each item maps to one process argument. A
string is also accepted and parsed with `shlex.split`. Registered commands run
with `shell=False`.

Supported placeholders are:

- `{context}`: path to the versioned Agent context manifest
- `{output_dir}`: host output directory
- `{instance_id}`: current benchmark instance ID
- `{tool_server_url}`: loopback HTTP tool endpoint
- `{system_prompt_path}`: rendered system prompt path
- `{instruction_path}`: rendered user instruction path
- `{tool_defs_path}`: tool definitions path
- `{task_context_path}`: task-specific context path
- `{config_dir}`: directory containing the selected config file

Unknown placeholders and malformed command values fail before process launch
with a configuration error. Relative `command` paths and `cwd` are interpreted
relative to the config file directory.

Agent-owned controls such as `--max-turns` are ordinary command arguments.
`timeout_sec` is the independent wall-clock limit enforced by `flyai-bench`.

## Runtime Flow

1. Load and merge the existing evaluation config, preserving top-level Agent
   registrations and the config file directory.
2. Resolve `--agent` or `agent.name`, validate the selected registration, and
   switch runtime mode to external.
3. Start the task container and loopback-only Tool Server.
4. Materialize prompts, `tool_defs.json`, task context, and
   `agent_context.json` under the task output directory.
5. Render the registered command placeholders and launch the Agent from its
   configured working directory.
6. Require `answer.json`, `answer.md`, plus configured extra output files;
   copy them into the container and run the existing verifier.

The manifest remains the stable integration protocol. It contains prompts,
tools, output paths, and non-secret model metadata. API keys are never written
to the manifest or substituted into process arguments.

## Environment Handling

Registered Agents receive a small execution environment containing basic
process variables (`PATH`, `HOME`, temporary-directory and locale variables),
the existing `LLM_*` values, and variables named in `pass_env`. The legacy
external-command path keeps its current inherited environment behavior.

`pass_env` contains names only. A listed variable must already exist in the
parent process; absent optional values are simply omitted. Invalid names are
rejected. This keeps secret handling explicit while avoiding command-line
leakage.

## Error Handling

Configuration errors identify the Agent and invalid field. Errors include an
unknown Agent name, unsupported protocol, empty command, invalid placeholder,
invalid working directory, unsafe output path, and invalid `pass_env` value.
Agent exit codes, timeouts, and missing outputs continue to be recorded in the
per-instance result. The result and original tool audit log are persisted before
the container is removed, including on failure paths.

## Compatibility

- Internal container Agents are unchanged when no registered Agent is selected.
- `answer.json` and `answer.md` remain mandatory.
- Additional output files remain constrained to `artifacts/`.
- Root compatibility copies and package copies remain byte-for-byte aligned.

## Testing

Unit tests cover config preservation, Agent selection, command rendering,
placeholder validation, relative working directories, environment allowlists,
legacy compatibility, and CLI parsing. Existing protocol tests cover manifest,
output paths, Tool Server safety, and sandbox wrappers. Smoke tests exercise
both dataset configs with a registered example Agent.

The OpenAI-compatible reference implementation is packaged as
`python3 -m flyai_bench.external_agent`. It can optionally record a declared
`artifacts/trajectory.jsonl`, and forces a final no-tool synthesis request after
its configured tool-turn budget so container-specific file-writing instructions
cannot trap a host Agent in an endless tool loop.
