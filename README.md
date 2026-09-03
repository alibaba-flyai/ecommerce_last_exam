# E-Commerce Last Exam

[![PyPI](https://img.shields.io/pypi/v/flyai-bench)](https://pypi.org/project/flyai-bench/)
![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)
![License](https://img.shields.io/badge/license-MIT-green)
[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/FlyaiLab/ecommerce_last_exam)
[![Leaderboard](https://img.shields.io/badge/HuggingFace-Leaderboard-orange)](https://huggingface.co/spaces/FlyaiLab/ecommerce_last_exam_leaderboard)

A benchmark for evaluating LLM agents on real-world travel scene and
e-commerce tasks. Each task runs in an isolated Docker container with
domain-specific CLI tools and databases. Agents must call tools, analyze
results, and produce structured answers. Scores range from `0.00` to `1.00`.

## Benchmark overview

| Config | Tasks | Domains |
|--------|-------|---------|
| `travel` | 77 | Hotel / Transport / Attraction |
| `e_commerce` | 43 | Travel gear / Food / Electronics / Lifestyle shopping |

**Total: 120 tasks** — Dataset: [FlyaiLab/ecommerce_last_exam](https://huggingface.co/datasets/FlyaiLab/ecommerce_last_exam)

## Prerequisites

- Python >= 3.10
- Docker (target platform `linux/amd64`)
- An OpenAI-compatible LLM endpoint for the agent and the judge/verifier
  (they can be the same endpoint)

## Installation

```bash
pip install flyai-bench
```

Or install from source:

```bash
git clone https://github.com/alibaba-flyai/ecommerce_last_exam
cd ecommerce_last_exam
pip install -e .
```

## Quick start

Run from a clone of this repository so the configuration template and
submission directories are available:

```bash
# 1. Copy the default config and fill in your LLM endpoint / API key
cp eval_config.yaml eval_config.local.yaml
# edit eval_config.local.yaml

# 2. Run evaluation (dry-run first to verify setup)
flyai-bench --config eval_config.local.yaml run --dataset-config travel --limit 5 --dry-run
flyai-bench --config eval_config.local.yaml run --dataset-config travel --limit 5

# 3. Check progress and generate a report
flyai-bench --config eval_config.local.yaml status
flyai-bench --config eval_config.local.yaml report
```

## Configuration

Copy `eval_config.yaml` to `eval_config.local.yaml` (already in `.gitignore`)
and fill in your values:

| Field | Meaning |
|-------|---------|
| `dataset.repo_id` / `dataset.config` / `dataset.split` | HuggingFace dataset, config (`travel` / `e_commerce`), and split |
| `docker.registry` | Image registry prefix. Leave empty for DockerHub (default). |
| `agent.name` | Select a named custom Agent from `agents`; omit it to use the built-in Agent |
| `agents.<name>.command` / `cwd` | Custom Agent startup argv and config-relative working directory |
| `agents.<name>.pass_env` | Host environment variable names explicitly passed to the custom Agent |
| `agents.<name>.output_paths` | Additional required files for that registered Agent; `agent.output_paths` is the shared default |
| `agent.llm_base_url` / `agent.llm_api_key` / `agent.llm_model` | OpenAI-compatible endpoint, API key, and model for the Agent |
| `verifier.judge_base_url` / `judge_api_key` / `judge_model` | Endpoint, key, and model for the judge/verifier |
| `runner.concurrency` / `runner.limit` / `runner.skip_done` | Parallelism, instance cap (`null` = all), skip completed |

> Agent credentials are exposed as `LLM_*` variables: inside the task container
> for the built-in Agent, or in the restricted host process environment for a
> registered Agent. A registered Agent can also read a host `LLM_API_KEY` when
> `agent.llm_api_key` is empty. Verifier credentials are passed only to the
> verifier. Never commit `eval_config.local.yaml`.

## Evaluation flow

```
HuggingFace Dataset
    │
    ▼
┌─────────────────────────────────┐
│  For each instance:             │
│  1. docker pull <image>         │
│  2. docker run (start env)      │
│  3. agent calls tools → answer  │
│  4. verifier (test.sh) scores   │
│  5. emit reward.txt (0 – 1)    │
└─────────────────────────────────┘
    │
    ▼
scores.jsonl + summary.json
```

## CLI commands

```bash
# Run evaluation
flyai-bench run [--dataset-config travel|e_commerce] [--limit N] [--dry-run]

# Multiple independent runs (pass@N)
flyai-bench run --dataset-config travel --runs 3

# Check progress
flyai-bench status

# Generate a report
flyai-bench report

# Package results for leaderboard submission
flyai-bench submit --model deepseek-v4-flash --provider deepseek
```

## Output

| File | Contents |
|------|----------|
| `scores.jsonl` | One line per instance: `{instance_id, domain, reward, duration_sec}` |
| `summary.json` | Aggregate statistics (`avg_reward`, breakdown `by_domain`) |

`reward` is a float in `0.00 – 1.00` produced by the verifier for each instance.

## Submitting results to the leaderboard

1. Run the evaluation and package results:

   ```bash
   flyai-bench --config eval_config.local.yaml run --dataset-config travel
   flyai-bench --config eval_config.local.yaml submit \
     --dataset-config travel \
     --model deepseek-v4-flash \
     --provider deepseek \
     --agent-type mini-swe-agent
   ```

2. Review `experiments/evaluation/travel/<slug>/metadata.yaml` and complete
   the model and agent details.

3. Open a pull request to [`alibaba-flyai/ecommerce_last_exam`](https://github.com/alibaba-flyai/ecommerce_last_exam).

4. CI validates the format automatically; once merged, the leaderboard updates.

### Submission package layout

```
experiments/evaluation/travel/20260830_deepseek-v4-flash_mini-swe-agent/
├── metadata.yaml      # model/agent info + evaluation statistics
├── scores.jsonl       # per instance: {instance_id, domain, reward, duration_sec}
└── summary.json       # aggregate statistics (avg_reward, by_domain)
```

## Integrating a custom Agent

Register your Agent once in the same YAML file used for the evaluation.
`flyai-bench` runs it on the host while owning the dataset, Docker sandbox, tool
execution, verifier, and result aggregation. Your Agent does not need to be
included in the benchmark image.

The recommended configuration passes one generated context manifest as a normal
command argument:

```yaml
agent:
  name: my-agent
  timeout_sec: 1800
  llm_base_url: "https://your-openai-compatible-endpoint/v1"
  llm_api_key: ""
  llm_model: "your-model"

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
      - artifacts/evidence/sources.json
    pass_env: []
```

Run one task before starting a full evaluation:

```bash
flyai-bench --config eval_config.local.yaml run \
  --agent my-agent \
  --dataset-config travel \
  --limit 1
```

`agent.name` selects the default registration; `--agent` overrides it for one
run. Relative `command` paths and `cwd` are resolved from the config file. The
command runs as an argv list with no shell expansion. Agent-specific controls,
such as `--max-turns`, belong directly in this command; `timeout_sec` is the
independent wall-clock limit enforced by `flyai-bench`.

For every task, the CLI writes an Agent manifest and substitutes its path for
`{context}`. The manifest contains:

```json
{
  "protocol_version": "1.0",
  "instance_id": "task-id",
  "prompts": {"system": "...", "user": "..."},
  "tools": {
    "url": "http://localhost:49152",
    "definitions": [{"type": "function", "function": {"name": "..."}}]
  },
  "task_context": {},
  "output": {
    "directory": "/absolute/path/to/eval_results/...",
    "paths": ["answer.json", "answer.md", "artifacts/evidence/sources.json"]
  },
  "llm": {
    "base_url": "https://your-endpoint/v1",
    "model": "your-model",
    "api_key_env": "LLM_API_KEY"
  }
}
```

Accept and read this single file in your Agent:

```python
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--context", required=True)
args = parser.parse_args()
context = json.loads(
    Path(args.context).read_text(encoding="utf-8")
)
```

Available command placeholders are `{context}`, `{output_dir}`,
`{instance_id}`, `{tool_server_url}`, `{system_prompt_path}`,
`{instruction_path}`, `{tool_defs_path}`, `{task_context_path}`, and
`{config_dir}`. API keys are intentionally not available as placeholders because
process arguments are visible to other local processes. Use `pass_env` for
provider-specific credentials.

Use `context["tools"]["definitions"]` as the LLM tool schemas. Execute a tool
with an HTTP request to:

```text
POST {context["tools"]["url"]}/call/{tool_name}
Content-Type: application/json

{"schemaPropertyName": "value"}
```

Tool argument names are passed through exactly as defined in the schema. The
task image must provide a non-empty `/app/tool_defs.json`; registered Agent
execution fails before launch when that contract is missing. Successful calls
are recorded by the original tools in `/app/.tool_calls.jsonl`, which is
consumed by the verifier and copied into the run artifacts.

The Agent must write every path listed in `context["output"]["paths"]` below the
declared output directory and exit with code `0`. `answer.json` and `answer.md`
are always required; additional files must live below `artifacts/` so they
cannot overwrite the verifier or sandbox. Configured paths are required. API
keys are never written into the manifest and remain available only through the
`LLM_API_KEY` environment variable or names explicitly listed in `pass_env`.

Each concurrent task receives its own manifest, tool port, and output directory.
Custom Agents must be declared under `agents` and selected with `agent.name` or
`--agent`; legacy mode and free-form command overrides are not supported.

See [`examples/external_agent.py`](examples/external_agent.py) for a complete
OpenAI-compatible function-calling loop. The same reference Agent is installed
as `python3 -m flyai_bench.external_agent`, which is the command used by the
`example-openai-agent` registration in `eval_config.yaml`.

## Project structure

```
ecommerce_last_exam/
├── README.md
├── pyproject.toml            # packaging (installs the `flyai-bench` CLI)
├── eval_config.yaml          # default config (copy to eval_config.local.yaml)
├── benchmark.yaml            # benchmark metadata
├── run_eval.py               # standalone evaluation script
├── agent.py                  # reference LLM agent
├── mini_swe_agent.py         # reference terminal agent
├── tool_server.py            # in-sandbox tool proxy (permission isolation)
├── sandbox_setup.sh          # container permission setup
├── validate_submission.py    # submission validation + leaderboard rebuild
├── experiments/              # submitted results + leaderboard.json
├── dataset_card/             # HuggingFace Dataset README + eval metadata
├── leaderboard_space/        # HuggingFace Space (leaderboard UI)
└── src/flyai_bench/          # installable package
```

## License

Released under the [MIT license](LICENSE).
