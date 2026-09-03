# E-Commerce Last Exam — Experiments

Evaluation results for the [E-Commerce Last Exam](https://huggingface.co/datasets/FlyaiLab/ecommerce_last_exam) benchmark.

## Leaderboard

The [official leaderboard](https://huggingface.co/spaces/FlyaiLab/ecommerce_last_exam_leaderboard)
contains the published baseline results. Community results submitted through
pull requests are listed below.

<!-- LEADERBOARD_START -->

_No community pull-request submissions yet._

<!-- LEADERBOARD_END -->

## How to Submit

### 1. Run Evaluation

```bash
pip install flyai-bench
cp eval_config.yaml eval_config.local.yaml
# Edit eval_config.local.yaml with your Agent and verifier endpoints.

# Run on travel config (77 tasks)
flyai-bench --config eval_config.local.yaml run --dataset-config travel

# Generate report
flyai-bench --config eval_config.local.yaml report --dataset-config travel
```

### 2. Package Results

```bash
flyai-bench --config eval_config.local.yaml submit \
  --dataset-config travel \
  --model your-model-name \
  --provider your-provider \
  --agent-type mini-swe-agent
```

### 3. Submit PR

1. Fork the [ecommerce_last_exam](https://github.com/alibaba-flyai/ecommerce_last_exam) repository
2. The `submit` command already places results in `experiments/evaluation/<config>/<slug>/`
3. Review and edit `metadata.yaml` with accurate model/agent details
4. Open a Pull Request to the main branch

CI will automatically validate the submission format. Once merged, the leaderboard updates automatically.

### Submission Format

Each submission lives in `experiments/evaluation/<config>/<date>_<model>_<agent>/`:

| File | Required | Description |
|------|----------|-------------|
| `metadata.yaml` | Yes | Model name, agent type, evaluation stats |
| `scores.jsonl` | Yes | Per-instance scores: `{instance_id, domain, reward, duration_sec}` |
| `summary.json` | No | Aggregate statistics (auto-generated if missing) |
| `trajs/` | No | Per-instance agent reasoning traces |

### metadata.yaml Schema

```yaml
model:
  name: "deepseek-v4-flash"
  provider: "deepseek"
  api_params:
    temperature: 0.0
    max_tokens: 8192
agent:
  type: "mini-swe-agent"
  version: "1.0"
  max_iterations: 30
  tool_calling: "terminal"
evaluation:
  benchmark: "ecommerce_last_exam"
  config: "travel"            # travel | e_commerce
  total_instances: 77
  completed: 75
  failed: 2
  avg_reward: 0.7234
  date: "2026-08-30"
notes: "Optional description"
```

## Benchmark Configs

| Config | Tasks | Domain |
|--------|-------|--------|
| `travel` | 77 | Hotel / Transport / Attraction |
| `e_commerce` | 43 | Travel gear / Food / Electronics / Lifestyle shopping |

## License

This project is released under the MIT license. See [LICENSE](../LICENSE) for details.
