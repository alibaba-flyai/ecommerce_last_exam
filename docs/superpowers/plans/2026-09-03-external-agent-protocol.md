# Registered Agent Protocol Implementation Record

> This plan records the first host-Agent implementation. Its environment-variable
> command interface was superseded by the named Agent registry and `manifest-v1`
> protocol described in
> [`2026-09-03-agent-registry-command-templates-design.md`](../specs/2026-09-03-agent-registry-command-templates-design.md).

## Final Architecture

The CLI owns the dataset, task container, loopback Tool Server, verifier, and
result aggregation. Users register a host-side Agent under `agents`, select it
with `agent.name` or `--agent`, and provide a startup command containing the
`{context}` placeholder. The substituted file is a versioned manifest containing
prompts, tool definitions, task context, output paths, and non-secret LLM
metadata.

Registered commands run with `shell=False` in a config-relative working
directory. Credentials are passed through a restricted process environment,
never through command placeholders or the manifest. The Agent must write
`answer.json`, `answer.md`, and any declared files below `artifacts/` before it
exits successfully.

## Completed Work

- [x] Preserve tool argument names exactly as defined by `tool_defs.json`.
- [x] Expose task tools through a loopback-only HTTP Tool Server.
- [x] Require `answer.json` and `answer.md` for every registered Agent run.
- [x] Support multiple safe relative output paths below `artifacts/`.
- [x] Persist the original `/app/.tool_calls.jsonl` audit log.
- [x] Add a versioned context manifest and safe command placeholders.
- [x] Add named Agent registrations and `--agent` selection.
- [x] Remove legacy `agent.mode`, `external_cmd`, `--agent-mode`, and
  `--agent-cmd` interfaces.
- [x] Package a reference OpenAI-compatible Agent and document the public
  integration flow for version `0.1.2`.

## Verification

Protocol tests cover command rendering, schema-preserving tool calls, manifest
contents, restricted environments, required and extra outputs, failure
artifacts, legacy-field rejection, and CLI selection. Smoke evaluations cover
both `travel` and `e_commerce` dataset configs.
