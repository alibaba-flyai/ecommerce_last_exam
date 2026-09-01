# flyai-bench

![Python](https://img.shields.io/badge/python-%3E%3D3.10-blue)
![License](https://img.shields.io/badge/license-MIT-green)

An evaluation framework for LLM agents. It loads evaluation tasks from a
HuggingFace dataset, runs an agent plus verifier inside isolated Docker
containers, and produces standardized scores (reward in the range `0.00 – 1.00`).

## Prerequisites

- Python >= 3.10
- Docker (with access to the image registry that hosts the benchmark images;
  the target platform is `linux/amd64`)
- An OpenAI-compatible LLM endpoint for the agent, and one for the judge/verifier
  (they can be the same endpoint)

## Installation

```bash
# From a clone of this repository
pip install -e .

# Or install the runtime dependencies directly
pip install pyyaml huggingface_hub openai
```

Installing the package exposes a `flyai-bench` console command. The examples
below also work by invoking `run_eval.py` directly.

## Quick start

```bash
# 1. Copy the default config and fill in your values
cp eval_config.yaml eval_config.local.yaml
#    edit eval_config.local.yaml: set the image registry, the LLM/judge
#    base URLs, API keys, and the model name

# 2. Run the evaluation
python run_eval.py run --config eval_config.local.yaml

# 3. Inspect results
python run_eval.py report --config eval_config.local.yaml
```

## Supported benchmarks

| Benchmark | Dataset | Tasks | Description |
|-----------|---------|-------|-------------|
| ecommerce_last_exam | [FlyaiLab/ecommerce_last_exam](https://huggingface.co/datasets/FlyaiLab/ecommerce_last_exam) | 1000 | Travel + e-commerce agent tool-use evaluation (500 travel, 500 e-commerce) |

## Configuration

Key fields in `eval_config.yaml` (copy to `eval_config.local.yaml` before editing):

| Field | Meaning |
|-------|---------|
| `dataset.repo_id` / `dataset.config` / `dataset.split` | HuggingFace dataset, config (`travel` / `e_commerce`), and split |
| `docker.registry` | Image registry that hosts the benchmark images. Set to your own registry, or override at runtime with `--registry`. Leave empty to use image names as-is. |
| `docker.platform` | Container platform (default `linux/amd64`) |
| `agent.cmd` | Command that runs the agent inside the container |
| `agent.llm_base_url` / `agent.llm_api_key` / `agent.llm_model` | OpenAI-compatible endpoint, key, and model for the agent |
| `verifier.judge_base_url` / `judge_api_key` / `judge_model` | Endpoint, key, and model for the judge/verifier |
| `runner.concurrency` / `runner.limit` / `runner.skip_done` | Parallelism, cap on number of instances (`null` = all), and whether to skip instances that already have a reward |

> API keys are read from your local config and injected into the containers as
> environment variables. Never commit `eval_config.local.yaml` — it is already
> in `.gitignore`.

## Evaluation flow

```
HuggingFace Dataset
    │
    ▼
┌─────────────────────────────────┐
│  For each instance:             │
│  1. docker pull <image>         │
│  2. docker run (start env)      │
│  3. agent calls tools -> answer │
│  4. verifier (test.sh) scores   │
│  5. emit reward.txt (0 – 1)     │
└─────────────────────────────────┘
    │
    ▼
scores.jsonl + summary.json
```

## CLI commands

```bash
# Run the evaluation
python run_eval.py run [--dataset-config travel|e_commerce] [--limit N] [--dry-run]

# Check progress
python run_eval.py status

# Generate a report
python run_eval.py report

# Package results for leaderboard submission
python run_eval.py submit --model deepseek-v4-flash --provider deepseek
```

## Output

| File | Contents |
|------|----------|
| `scores.jsonl` | One line per instance: `{instance_id, domain, reward, duration_sec}` |
| `summary.json` | Aggregate statistics (`avg_reward`, breakdown `by_domain`) |

`reward` is a float in `0.00 – 1.00` produced by the verifier for each instance.

## Submitting results to the leaderboard

1. After evaluating, package the results with `submit`:

   ```bash
   python run_eval.py submit \
     --dataset-config travel \
     --model deepseek-v4-flash \
     --provider deepseek \
     --agent-type mini-swe-agent
   ```

2. Review `experiments/evaluation/travel/<slug>/metadata.yaml` and complete the
   model and agent details.

3. Open a pull request to the `main` branch of the flyai-bench repository.

4. CI validates the format automatically; once merged, the leaderboard updates.

### Submission package layout

```
evaluation/travel/20260830_deepseek-v4-flash_mini-swe-agent/
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
python run_eval.py run --agent-cmd "python /app/my_agent.py"
```

`agent.py` (a minimal LLM agent) and `mini_swe_agent.py` (a terminal/bash agent)
are provided as reference implementations.

## Project structure

```
flyai-bench/
├── README.md
├── pyproject.toml            # packaging (installs the `flyai-bench` CLI)
├── run_eval.py               # evaluation CLI (run / status / report / submit)
├── eval_config.yaml          # default config (copy to eval_config.local.yaml)
├── benchmark.yaml            # benchmark metadata
├── agent.py                  # reference LLM agent
├── mini_swe_agent.py         # reference terminal agent
├── tool_server.py            # in-sandbox tool proxy (permission isolation)
├── sandbox_setup.sh          # container permission setup
├── validate_submission.py    # submission validation + leaderboard rebuild
├── experiments/              # submitted results + leaderboard.json
├── leaderboard_space/        # HuggingFace Space (leaderboard UI)
└── src/flyai_bench/          # installable package mirror of the above
```

> Note: the top-level scripts and the `src/flyai_bench/` package currently hold
> parallel copies of the same code. Prefer editing one and keeping them in sync
> (or consolidating on the package) to avoid drift.

## License

Released under the MIT license. See [LICENSE](LICENSE) for details.
