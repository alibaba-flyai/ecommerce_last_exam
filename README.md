# E-Commerce Last Exam

[![PyPI](https://img.shields.io/pypi/v/flyai-bench)](https://pypi.org/project/flyai-bench/)
![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)
![License](https://img.shields.io/badge/license-MIT-green)
[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/FlyaiLab/ecommerce_last_exam)
[![Leaderboard](https://img.shields.io/badge/HuggingFace-Leaderboard-orange)](https://huggingface.co/spaces/FlyaiLab/ecommerce_last_exam_leaderboard)

A benchmark for evaluating LLM agents on real-world travel planning and
e-commerce tasks. Each task runs in an isolated Docker container with
domain-specific CLI tools and databases. Agents must call tools, analyze
results, and produce structured answers. Scores range from `0.00` to `1.00`.

## Benchmark overview

| Config | Tasks | Domains |
|--------|-------|---------|
| `travel` | 77 | Hotel / Transport / Attraction trip planning |
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

```bash
# 1. Copy the default config and fill in your LLM endpoint / API key
cp eval_config.yaml eval_config.local.yaml
# edit eval_config.local.yaml

# 2. Run evaluation (dry-run first to verify setup)
flyai-bench run --dataset-config travel --limit 5 --dry-run
flyai-bench run --dataset-config travel --limit 5

# 3. Check progress and generate a report
flyai-bench status
flyai-bench report
```

## Configuration

Copy `eval_config.yaml` to `eval_config.local.yaml` (already in `.gitignore`)
and fill in your values:

| Field | Meaning |
|-------|---------|
| `dataset.repo_id` / `dataset.config` / `dataset.split` | HuggingFace dataset, config (`travel` / `e_commerce`), and split |
| `docker.registry` | Image registry prefix. Leave empty for DockerHub (default). |
| `agent.llm_base_url` / `agent.llm_api_key` / `agent.llm_model` | OpenAI-compatible endpoint, API key, and model for the agent |
| `verifier.judge_base_url` / `judge_api_key` / `judge_model` | Endpoint, key, and model for the judge/verifier |
| `runner.concurrency` / `runner.limit` / `runner.skip_done` | Parallelism, instance cap (`null` = all), skip completed |

> API keys are read from your local config and injected into containers as
> environment variables. Never commit `eval_config.local.yaml`.

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
   flyai-bench run --dataset-config travel
   flyai-bench submit \
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

## Writing a custom agent

Implement a script that, inside the container:

1. Reads `/app/tool_defs.json` for the tool definitions
2. Reads `/app/system.md` + `/app/instruction.md` for the task
3. Calls `/app/tools/<name> --arg val` to execute a tool
4. Writes the result to `/app/answer.json`

Point the runner at your agent with `--agent-cmd`:

```bash
flyai-bench run --agent-cmd "python /app/my_agent.py"
```

`agent.py` (a minimal LLM agent) and `mini_swe_agent.py` (a terminal/bash agent)
are provided as reference implementations.

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
├── leaderboard_space/        # HuggingFace Space (leaderboard UI)
└── src/flyai_bench/          # installable package
```

## License

Released under the [MIT license](LICENSE).
