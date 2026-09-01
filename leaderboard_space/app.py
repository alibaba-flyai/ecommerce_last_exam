"""E-Commerce Last Exam Leaderboard — HuggingFace Space (Gradio)."""
import json
from pathlib import Path

import gradio as gr
import pandas as pd

LEADERBOARD_FILE = Path(__file__).parent / "leaderboard.json"

ABOUT_TEXT = """
## E-Commerce Last Exam

A benchmark for evaluating LLM agents on real-world travel planning and e-commerce tasks.
Each task runs in a Docker container with domain-specific tools and databases.
Agents must call tools via CLI, analyze results, and produce a structured answer.

**Dataset**: [FlyaiLab/ecommerce_last_exam](https://huggingface.co/datasets/FlyaiLab/ecommerce_last_exam)
**Evaluation CLI**: [flyai-bench](https://github.com/alibaba-flyai/ecommerce_last_exam)

### Configs
| Config | Tasks | Language | Domains |
|--------|-------|----------|---------|
| `travel` | 77 | Chinese | Hotel / Transport / Attraction |
| `e_commerce` | 43 | English | Travel gear / Food / Electronics / Lifestyle |

### Scoring
Each task is scored 0.00–1.00 combining hard checks (rubrics.py) and LLM soft judge (judge.py).
"""

SUBMIT_TEXT = """
## How to Submit

```bash
pip install flyai-bench
flyai-bench run --dataset-config travel
flyai-bench report --dataset-config travel
flyai-bench submit --model your-model --provider your-provider
```

Then open a PR to the [flyai-bench](https://github.com/alibaba-flyai/ecommerce_last_exam) repository
with your results in `experiments/evaluation/<config>/<slug>/`.

CI validates the submission automatically. Once merged, the leaderboard updates.

### Required Files
| File | Description |
|------|-------------|
| `metadata.yaml` | Model/agent info + evaluation stats |
| `scores.jsonl` | Per-instance: `{instance_id, domain, reward, duration_sec}` |
| `summary.json` | Aggregate statistics (auto-generated if missing) |
"""


def load_leaderboard():
    if not LEADERBOARD_FILE.exists():
        return {"configs": {"travel": [], "e_commerce": []}}
    with open(LEADERBOARD_FILE, encoding="utf-8") as f:
        return json.load(f)


def make_dataframe(entries):
    if not entries:
        return pd.DataFrame(columns=["Rank", "Model", "Provider", "Agent", "Avg Reward", "Completed", "Date"])
    rows = []
    for i, e in enumerate(sorted(entries, key=lambda x: -x.get("avg_reward", 0)), 1):
        rows.append({
            "Rank": i,
            "Model": e.get("model", ""),
            "Provider": e.get("provider", ""),
            "Agent": e.get("agent", ""),
            "Avg Reward": round(e.get("avg_reward", 0), 4),
            "Completed": f"{e.get('completed', 0)}/{e.get('total', 0)}",
            "Date": e.get("date", ""),
        })
    return pd.DataFrame(rows)


def get_travel_df():
    lb = load_leaderboard()
    return make_dataframe(lb.get("configs", {}).get("travel", []))


def get_ecommerce_df():
    lb = load_leaderboard()
    return make_dataframe(lb.get("configs", {}).get("e_commerce", []))


with gr.Blocks(title="E-Commerce Last Exam Leaderboard") as demo:
    gr.Markdown("# E-Commerce Last Exam Leaderboard")

    with gr.Tabs():
        with gr.Tab("Travel (77 tasks)"):
            gr.Markdown("Chinese travel planning tasks — hotel, transport, attraction domains")
            travel_table = gr.Dataframe(
                value=get_travel_df,
                headers=["Rank", "Model", "Provider", "Agent", "Avg Reward", "Completed", "Date"],
                interactive=False,
                every=300,
            )

        with gr.Tab("E-Commerce (43 tasks)"):
            gr.Markdown("English shopping/consumption tasks — gear, food, electronics, lifestyle domains")
            ecom_table = gr.Dataframe(
                value=get_ecommerce_df,
                headers=["Rank", "Model", "Provider", "Agent", "Avg Reward", "Completed", "Date"],
                interactive=False,
                every=300,
            )

        with gr.Tab("About"):
            gr.Markdown(ABOUT_TEXT)

        with gr.Tab("Submit"):
            gr.Markdown(SUBMIT_TEXT)


if __name__ == "__main__":
    demo.launch()
