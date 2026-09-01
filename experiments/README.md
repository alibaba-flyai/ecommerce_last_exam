# E-Commerce Last Exam — Experiments

Evaluation results for the [E-Commerce Last Exam](https://huggingface.co/datasets/FlyaiLab/ecommerce_last_exam) benchmark.

## Leaderboard

<!-- LEADERBOARD_START -->

_No submissions yet. Be the first to submit!_

<!-- LEADERBOARD_END -->

## How to Submit

### 1. Run Evaluation

```bash
git clone https://github.com/alibaba-flyai/ecommerce_last_exam
cd flyai-bench
pip install pyyaml huggingface_hub openai

# Run on travel config (500 tasks)
python run_eval.py run --dataset-config travel

# Generate report
python run_eval.py report --dataset-config travel
```

### 2. Package Results

```bash
python run_eval.py submit \
  --dataset-config travel \
  --model your-model-name \
  --provider your-provider \
  --agent-type mini-swe-agent
```

### 3. Submit PR

1. Fork the [flyai-bench](https://github.com/alibaba-flyai/ecommerce_last_exam) repository
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
  total_instances: 500
  completed: 498
  failed: 2
  avg_reward: 0.7234
  date: "2026-08-30"
notes: "Optional description"
```

## Benchmark Configs

| Config | Tasks | Language | Domain |
|--------|-------|----------|--------|
| `travel` | 500 | Chinese | Hotel / Traffic / POI trip planning |
| `e_commerce` | 500 | English | Fashion / Health / Electronics shopping |

## License

This project is released under the MIT license. See [LICENSE](../LICENSE) for details.
