#!/usr/bin/env python3
"""Validate submission format and update the leaderboard.

Usage:
  # Validate a single submission
  python validate_submission.py check evaluation/travel/20260830_deepseek-v4_mini-swe-agent

  # Validate all new submissions in a PR
  python validate_submission.py check-pr

  # Rebuild the leaderboard
  python validate_submission.py rebuild-leaderboard
"""
import argparse
import json
import os
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

VALID_CONFIGS = {"travel", "e_commerce"}

METADATA_REQUIRED = {
    "model": {"name"},
    "agent": {"type"},
    "evaluation": {"benchmark", "config", "total_instances", "completed", "avg_reward", "date"},
}


def load_yaml_or_json(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if yaml:
        return yaml.safe_load(text)
    return json.loads(text)


def validate_metadata(meta_path):
    errors = []
    try:
        meta = load_yaml_or_json(meta_path)
    except Exception as e:
        return [f"metadata.yaml parse error: {e}"]

    for section, fields in METADATA_REQUIRED.items():
        if section not in meta:
            errors.append(f"missing section: {section}")
            continue
        for field in fields:
            if field not in meta[section]:
                errors.append(f"missing field: {section}.{field}")

    if meta.get("evaluation", {}).get("config") not in VALID_CONFIGS:
        errors.append(f"evaluation.config must be one of {VALID_CONFIGS}")

    avg = meta.get("evaluation", {}).get("avg_reward")
    if avg is not None and not (0 <= avg <= 1):
        errors.append(f"evaluation.avg_reward must be in [0, 1], got {avg}")

    runs = meta.get("evaluation", {}).get("runs")
    if runs is not None and (not isinstance(runs, int) or runs < 1):
        errors.append(f"evaluation.runs must be a positive integer, got {runs}")

    return errors


def validate_scores(scores_path, dataset_ids=None):
    errors = []
    warnings = []
    if not os.path.exists(scores_path):
        return ["scores.jsonl not found"]

    records = []
    with open(scores_path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"scores.jsonl line {i}: invalid JSON: {e}")
                continue
            if "instance_id" not in rec:
                errors.append(f"scores.jsonl line {i}: missing instance_id")
                continue
            reward = rec.get("reward")
            if reward is not None and not (0 <= reward <= 1):
                errors.append(f"scores.jsonl line {i}: reward {reward} not in [0, 1]")
            records.append(rec)

    if not records:
        errors.append("scores.jsonl is empty")

    has_run_id = any(r.get("run_id") is not None for r in records)
    if has_run_id:
        from collections import Counter
        runs_per_instance = Counter(r["instance_id"] for r in records)
        run_counts = set(runs_per_instance.values())
        if len(run_counts) > 1:
            warnings.append(f"inconsistent run counts per instance: {run_counts}")

    if dataset_ids:
        submitted_ids = {r["instance_id"] for r in records}
        unknown = submitted_ids - dataset_ids
        if unknown:
            errors.append(f"{len(unknown)} unknown instance_ids not in dataset (first 3: {list(unknown)[:3]})")

    for w in warnings:
        print(f"WARNING: {w}", file=sys.stderr)

    return errors


def validate_submission(submit_dir, dataset_ids=None):
    submit_dir = Path(submit_dir)
    errors = []

    if not submit_dir.is_dir():
        return [f"not a directory: {submit_dir}"]

    meta_path = submit_dir / "metadata.yaml"
    if not meta_path.exists():
        meta_path = submit_dir / "metadata.json"
    if not meta_path.exists():
        errors.append("missing metadata.yaml")
    else:
        errors.extend(validate_metadata(str(meta_path)))

    scores_path = submit_dir / "scores.jsonl"
    errors.extend(validate_scores(str(scores_path), dataset_ids))

    return errors


def find_submissions(eval_root="experiments/evaluation"):
    subs = []
    eval_path = Path(eval_root)
    if not eval_path.is_dir():
        return subs
    for config_dir in eval_path.iterdir():
        if not config_dir.is_dir():
            continue
        for run_dir in config_dir.iterdir():
            if not run_dir.is_dir():
                continue
            if (run_dir / "scores.jsonl").exists() or (run_dir / "metadata.yaml").exists():
                subs.append(run_dir)
    return sorted(subs)


def load_submission_entry(submit_dir):
    submit_dir = Path(submit_dir)
    meta_path = submit_dir / "metadata.yaml"
    if not meta_path.exists():
        meta_path = submit_dir / "metadata.json"

    meta = load_yaml_or_json(str(meta_path))

    scores = []
    scores_path = submit_dir / "scores.jsonl"
    with open(scores_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                scores.append(json.loads(line))

    valid = [s for s in scores if s.get("reward") is not None]
    has_run_id = any(s.get("run_id") is not None for s in valid)

    if has_run_id:
        from collections import defaultdict
        by_instance = defaultdict(list)
        by_run = defaultdict(list)
        for s in valid:
            by_instance[s["instance_id"]].append(s["reward"])
            by_run[s["run_id"]].append(s["reward"])

        instance_means = [sum(rs) / len(rs) for rs in by_instance.values()]
        avg_reward = round(sum(instance_means) / len(instance_means), 4) if instance_means else 0
        completed = len(by_instance)

        run_ids = sorted(by_run.keys())
        runs = len(run_ids)
        run_scores = [round(sum(by_run[rid]) / len(by_run[rid]), 4) for rid in run_ids]

        std = 0.0
        if runs > 1:
            mean_of_runs = sum(run_scores) / len(run_scores)
            variance = sum((s - mean_of_runs) ** 2 for s in run_scores) / (len(run_scores) - 1)
            std = round(variance ** 0.5, 4)

        total = len(set(s["instance_id"] for s in scores))
    else:
        avg_reward = round(sum(s["reward"] for s in valid) / len(valid), 4) if valid else 0
        completed = len(valid)
        total = len(scores)
        runs = 1
        run_scores = []
        std = 0.0

    domain_scores = {}
    for s in valid:
        d = s.get("domain", "unknown")
        domain_scores.setdefault(d, []).append(s["reward"])
    by_domain = {}
    for d, rs in sorted(domain_scores.items()):
        by_domain[d] = round(sum(rs) / len(rs), 4)

    return {
        "slug": submit_dir.name,
        "config": meta.get("evaluation", {}).get("config", submit_dir.parent.name),
        "model": meta.get("model", {}).get("name", "unknown"),
        "provider": meta.get("model", {}).get("provider", ""),
        "agent": meta.get("agent", {}).get("type", "unknown"),
        "total": total,
        "completed": completed,
        "avg_reward": avg_reward,
        "std": std,
        "runs": runs,
        "run_scores": run_scores,
        "by_domain": by_domain,
        "date": meta.get("evaluation", {}).get("date", ""),
        "notes": meta.get("notes", ""),
    }


def rebuild_leaderboard(eval_root="experiments/evaluation", output="experiments/leaderboard.json"):
    subs = find_submissions(eval_root)
    entries = []
    for sub_dir in subs:
        try:
            entry = load_submission_entry(sub_dir)
            entries.append(entry)
        except Exception as e:
            print(f"WARNING: skip {sub_dir}: {e}", file=sys.stderr)

    entries.sort(key=lambda e: -e["avg_reward"])

    leaderboard = {
        "benchmark": "ecommerce_last_exam",
        "updated": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M"),
        "configs": {},
    }
    for config in VALID_CONFIGS:
        config_entries = [e for e in entries if e["config"] == config]
        leaderboard["configs"][config] = config_entries

    with open(output, "w", encoding="utf-8") as f:
        json.dump(leaderboard, f, indent=2, ensure_ascii=False)

    print(f"Leaderboard written to {output} ({len(entries)} entries)")
    return leaderboard


def render_readme_table(leaderboard):
    lines = []
    for config in sorted(leaderboard["configs"]):
        entries = leaderboard["configs"][config]
        if not entries:
            continue
        has_runs = any(e.get("runs", 1) > 1 for e in entries)
        lines.append(f"\n### {config.replace('_', ' ').title()}\n")
        header = "| Rank | Model | Agent | Avg Reward |"
        sep = "|------|-------|-------|-----------|"
        if has_runs:
            header += " Std | Runs |"
            sep += "-----|------|"
        header += " Completed | Date |"
        sep += "-----------|------|"
        lines.append(header)
        lines.append(sep)
        for i, e in enumerate(entries, 1):
            row = f"| {i} | {e['model']} | {e['agent']} | **{e['avg_reward']:.4f}** |"
            if has_runs:
                row += f" ±{e.get('std', 0):.4f} | {e.get('runs', 1)} |"
            row += f" {e['completed']}/{e['total']} | {e['date']} |"
            lines.append(row)
    return "\n".join(lines)


def cmd_check(args):
    errors = validate_submission(args.submission_dir)
    if errors:
        print(f"FAIL: {args.submission_dir}")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print(f"OK: {args.submission_dir}")


def cmd_check_pr(args):
    import subprocess
    r = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=A", "origin/main...HEAD"],
        capture_output=True, text=True)
    new_files = r.stdout.strip().split("\n") if r.stdout.strip() else []

    submission_dirs = set()
    for f in new_files:
        if f.startswith("experiments/evaluation/"):
            parts = Path(f).parts
            if len(parts) >= 4:
                submission_dirs.add(str(Path(*parts[:4])))

    if not submission_dirs:
        print("No new submissions found in PR")
        return

    all_ok = True
    for sub_dir in sorted(submission_dirs):
        errors = validate_submission(sub_dir)
        if errors:
            print(f"FAIL: {sub_dir}")
            for e in errors:
                print(f"  - {e}")
            all_ok = False
        else:
            print(f"OK: {sub_dir}")

    if not all_ok:
        sys.exit(1)


def cmd_rebuild(args):
    lb = rebuild_leaderboard(args.eval_root, args.output)
    table = render_readme_table(lb)
    print(table)

    readme_path = Path("experiments/README.md")
    if readme_path.exists():
        content = readme_path.read_text(encoding="utf-8")
        marker_start = "<!-- LEADERBOARD_START -->"
        marker_end = "<!-- LEADERBOARD_END -->"
        if marker_start in content and marker_end in content:
            before = content[:content.index(marker_start) + len(marker_start)]
            after = content[content.index(marker_end):]
            new_content = before + "\n" + table + "\n" + after
            readme_path.write_text(new_content, encoding="utf-8")
            print(f"\nREADME.md leaderboard section updated")


def main():
    parser = argparse.ArgumentParser(description="E-Commerce Last Exam submission validator")
    sub = parser.add_subparsers(dest="command")

    p_check = sub.add_parser("check", help="Validate a single submission directory")
    p_check.add_argument("submission_dir", help="Submission directory path")

    p_pr = sub.add_parser("check-pr", help="Validate all new submissions in a PR")

    p_rebuild = sub.add_parser("rebuild-leaderboard", help="Rebuild the leaderboard")
    p_rebuild.add_argument("--eval-root", default="experiments/evaluation")
    p_rebuild.add_argument("--output", default="experiments/leaderboard.json")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "check":
        cmd_check(args)
    elif args.command == "check-pr":
        cmd_check_pr(args)
    elif args.command == "rebuild-leaderboard":
        cmd_rebuild(args)


if __name__ == "__main__":
    main()
