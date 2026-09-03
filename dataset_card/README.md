---
license: mit
tags:
  - benchmark
linked_spaces:
  - FlyaiLab/ecommerce_last_exam_leaderboard
dataset_info:
  features:
    - name: instance_id
      dtype: string
    - name: domain
      dtype: string
    - name: config
      dtype: string
    - name: user_question
      dtype: string
    - name: docker_image
      dtype: string
  splits:
    - name: test
      num_examples: 120
configs:
  - config_name: default
    data_files:
      - split: test
        path: data/test-*
  - config_name: travel
    data_files:
      - split: test
        path: data/travel/test-*
  - config_name: e_commerce
    data_files:
      - split: test
        path: data/e_commerce/test-*
---

# E-Commerce Last Exam

<iframe
  src="https://FlyaiLab-ecommerce_last_exam_leaderboard.hf.space"
  frameborder="0"
  width="100%"
  height="600"
></iframe>

A benchmark for evaluating LLM agents on **120 real-world travel scene and
e-commerce tool-use tasks**. Each task runs in an isolated Docker container
with domain-specific CLI tools and SQLite databases. Agents must search,
analyze, and produce structured recommendations.

- **Repository**: [alibaba-flyai/ecommerce_last_exam](https://github.com/alibaba-flyai/ecommerce_last_exam)
- **Evaluation CLI**: [flyai-bench](https://pypi.org/project/flyai-bench/) (`pip install flyai-bench`)
- **Leaderboard**: [FlyaiLab/ecommerce_last_exam_leaderboard](https://huggingface.co/spaces/FlyaiLab/ecommerce_last_exam_leaderboard)

## Dataset Summary

E-Commerce Last Exam consists of 120 tasks across two configs:

| Config | Tasks | Domains |
|--------|-------|---------|
| `travel` | 77 | Hotel booking, transport routing, attraction planning |
| `e_commerce` | 43 | Travel gear, food, electronics, lifestyle shopping |

Each task provides a natural-language user question and a pre-built Docker
image containing the environment, CLI tools, databases, and test harness.
Agents interact with the environment through tool calls and produce a
structured answer scored from `0.00` to `1.00`.

## Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `instance_id` | string | Unique task identifier, such as `attraction_auckland_extreme_sports_415` |
| `domain` | string | Task domain, such as `attraction`, `hotel`, `transport`, or `consume` |
| `config` | string | Benchmark config: `travel` or `e_commerce` |
| `user_question` | string | Natural-language task description |
| `docker_image` | string | Docker image for the task environment |

## Usage

```python
from datasets import load_dataset

# Load all 120 tasks.
dataset = load_dataset("FlyaiLab/ecommerce_last_exam", split="test")

# Load one benchmark config.
travel = load_dataset(
    "FlyaiLab/ecommerce_last_exam", "travel", split="test"
)
ecommerce = load_dataset(
    "FlyaiLab/ecommerce_last_exam", "e_commerce", split="test"
)
```

## Evaluation

Clone the repository so the configuration template and submission directories
are available:

```bash
git clone https://github.com/alibaba-flyai/ecommerce_last_exam.git
cd ecommerce_last_exam
pip install flyai-bench
cp eval_config.yaml eval_config.local.yaml
# Edit eval_config.local.yaml with your Agent and verifier endpoints.

flyai-bench --config eval_config.local.yaml run \
  --dataset-config travel \
  --limit 5 \
  --dry-run
flyai-bench --config eval_config.local.yaml run --dataset-config travel
flyai-bench --config eval_config.local.yaml report
```

The CLI supports its built-in container Agent and named custom Agents registered
in the same YAML file. See the
[custom Agent integration guide](https://github.com/alibaba-flyai/ecommerce_last_exam#integrating-a-custom-agent)
for the `manifest-v1` protocol and submission workflow.

## Citation

```bibtex
@misc{ecommerce_last_exam,
  title={E-Commerce Last Exam: A Benchmark for LLM Agent Evaluation on Real-World Tool-Use Tasks},
  author={FlyaiLab},
  year={2026},
  url={https://huggingface.co/datasets/FlyaiLab/ecommerce_last_exam}
}
```
