#!/usr/bin/env python3
"""E-Commerce Last Exam evaluation CLI.

Usage:
  flyai-bench run                                    # run with default config
  flyai-bench run --config eval_config.local.yaml    # custom config
  flyai-bench run --dataset-config e_commerce --limit 10
  flyai-bench run --dry-run
  flyai-bench status
  flyai-bench report
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


# ─── Config ───────────────────────────────────────────────────────────────────

DEFAULT_CONFIG = {
    "dataset": {
        "repo_id": "FlyaiLab/ecommerce_last_exam",
        "config": "travel",
        "split": "test",
    },
    "docker": {
        "registry": "",
        "platform": "linux/amd64",
    },
    "agent": {
        "mode": "internal",
        "cmd": "python3 /app/mini_swe_agent.py",
        "external_cmd": "",
        "timeout_sec": 1800,
        "llm_base_url": "http://host.docker.internal:4000",
        "llm_api_key": "",
        "llm_model": "qwen3.7-plus",
    },
    "verifier": {
        "timeout_sec": 1200,
        "judge_base_url": "http://host.docker.internal:4000",
        "judge_api_key": "",
        "judge_model": "qwen3.7-plus",
    },
    "runner": {
        "concurrency": 1,
        "output_dir": "./eval_results",
        "limit": None,
        "skip_done": True,
    },
}


def load_config(path):
    if not path or not os.path.exists(path):
        return DEFAULT_CONFIG.copy()
    if yaml is None:
        print("WARNING: pyyaml not installed, using default config", file=sys.stderr)
        return DEFAULT_CONFIG.copy()
    with open(path, encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    cfg = DEFAULT_CONFIG.copy()
    for section in cfg:
        if section in user_cfg and isinstance(user_cfg[section], dict):
            cfg[section] = {**cfg[section], **user_cfg[section]}
    return cfg


def apply_overrides(cfg, args):
    if getattr(args, "dataset_config", None):
        cfg["dataset"]["config"] = args.dataset_config
    if getattr(args, "limit", None) is not None:
        cfg["runner"]["limit"] = args.limit
    if getattr(args, "concurrency", None) is not None:
        cfg["runner"]["concurrency"] = args.concurrency
    if getattr(args, "output_dir", None):
        cfg["runner"]["output_dir"] = args.output_dir
    if getattr(args, "agent_mode", None):
        cfg["agent"]["mode"] = args.agent_mode
    if getattr(args, "agent_cmd", None):
        if cfg["agent"]["mode"] == "external":
            cfg["agent"]["external_cmd"] = args.agent_cmd
        else:
            cfg["agent"]["cmd"] = args.agent_cmd
    if getattr(args, "registry", None):
        cfg["docker"]["registry"] = args.registry
    return cfg


# ─── Dataset ──────────────────────────────────────────────────────────────────

def load_instances(cfg):
    repo_id = cfg["dataset"]["repo_id"]
    config = cfg["dataset"]["config"]
    split = cfg["dataset"]["split"]
    try:
        from datasets import load_dataset
        ds = load_dataset(repo_id, config, split=split)
        return [dict(row) for row in ds]
    except ImportError:
        pass
    try:
        import pandas as pd
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id, f"data/{config}/{split}-00000-of-00001.parquet",
                               repo_type="dataset")
        df = pd.read_parquet(path)
        return df.to_dict(orient="records")
    except Exception:
        pass
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id, f"data/{config}/{split}.jsonl", repo_type="dataset")
    instances = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                instances.append(json.loads(line))
    return instances


# ─── Docker helpers ───────────────────────────────────────────────────────────

def resolve_image(docker_image, registry):
    if registry and "/" in docker_image and not docker_image.startswith(registry):
        return f"{registry}/{docker_image}"
    return docker_image


def docker_pull(image, timeout=300):
    r = subprocess.run(["docker", "pull", image],
                       capture_output=True, text=True, timeout=timeout)
    return r.returncode == 0, r.stderr[-300:] if r.returncode != 0 else ""


def docker_run(image, name, platform, env_vars=None, timeout_sleep=3600):
    cmd = ["docker", "run", "-d", "--name", name, "--platform", platform]
    if env_vars:
        for k, v in env_vars.items():
            if v:
                cmd += ["-e", f"{k}={v}"]
    cmd += [image, "sleep", str(timeout_sleep)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    return r.returncode == 0, r.stdout.strip(), r.stderr[-200:]


def docker_exec(name, cmd_str, timeout=1800):
    cmd = ["docker", "exec", name, "bash", "-c", cmd_str]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr


def docker_cp_from(name, container_path, local_path):
    r = subprocess.run(["docker", "cp", f"{name}:{container_path}", local_path],
                       capture_output=True, text=True, timeout=30)
    return r.returncode == 0


def docker_rm(name):
    subprocess.run(["docker", "rm", "-f", name],
                   capture_output=True, text=True, timeout=30)


def docker_port(name, container_port=9999):
    r = subprocess.run(["docker", "port", name, str(container_port)],
                       capture_output=True, text=True, timeout=10)
    if r.returncode != 0:
        return None
    mapping = r.stdout.strip().split("\n")[0]
    return mapping.split(":")[-1]


# ─── Evaluation logic ─────────────────────────────────────────────────────────

def eval_instance(instance, cfg, run_id=None):
    """Evaluate a single instance: one container, agent runs as non-root (cannot read tests/ or trip.db)."""
    instance_id = instance["instance_id"]
    registry = cfg["docker"]["registry"]
    platform = cfg["docker"]["platform"]
    if run_id is not None:
        output_dir = os.path.join(cfg["runner"]["output_dir"], cfg["dataset"]["config"],
                                  instance_id, f"run_{run_id}")
    else:
        output_dir = os.path.join(cfg["runner"]["output_dir"], cfg["dataset"]["config"], instance_id)
    os.makedirs(output_dir, exist_ok=True)

    result = {"instance_id": instance_id, "domain": instance.get("domain", "")}
    if run_id is not None:
        result["run_id"] = run_id
    t0 = time.time()
    suffix = f"_r{run_id}" if run_id is not None else ""
    container_name = f"eval_{instance_id[:12]}_{int(t0) % 10000}{suffix}"

    is_external = cfg["agent"].get("mode") == "external"

    try:
        # 1. Pull
        image = resolve_image(instance["docker_image"], registry)
        ok, err = docker_pull(image)
        if not ok:
            result.update(reward=None, error=f"pull failed: {err}", phase="pull")
            return result

        # 2. Start the container (root) and set up permission isolation
        all_env = {
            "LLM_BASE_URL": cfg["agent"]["llm_base_url"],
            "LLM_API_KEY": cfg["agent"]["llm_api_key"],
            "LLM_MODEL": cfg["agent"]["llm_model"],
            "JUDGE_BASE_URL": cfg["verifier"]["judge_base_url"],
            "JUDGE_API_KEY": cfg["verifier"]["judge_api_key"],
            "JUDGE_MODEL": cfg["verifier"]["judge_model"],
        }
        if is_external:
            all_env["AGENT_MODE"] = "external"
        run_cmd = ["docker", "run", "-d", "--name", container_name, "--platform", platform]
        if is_external:
            run_cmd += ["-p", "0:9999"]
        for k, v in all_env.items():
            if v:
                run_cmd += ["-e", f"{k}={v}"]
        run_cmd += [image, "sleep", str(cfg["agent"]["timeout_sec"] + cfg["verifier"]["timeout_sec"] + 120)]
        r = subprocess.run(run_cmd, capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            result.update(reward=None, error=f"start failed: {r.stderr[-200:]}", phase="start")
            return result

        # 3. Inject sandbox script + tool_server + agent, set up permission isolation
        script_dir = os.path.dirname(os.path.abspath(__file__))
        subprocess.run(["docker", "cp", os.path.join(script_dir, "tool_server.py"),
                        f"{container_name}:/app/.tool_server.py"], capture_output=True, timeout=10)
        subprocess.run(["docker", "cp", os.path.join(script_dir, "sandbox_setup.sh"),
                        f"{container_name}:/app/.sandbox_setup.sh"], capture_output=True, timeout=10)
        subprocess.run(["docker", "cp", os.path.join(script_dir, "mini_swe_agent.py"),
                        f"{container_name}:/app/mini_swe_agent.py"], capture_output=True, timeout=10)
        code, _, stderr = docker_exec(container_name, "bash /app/.sandbox_setup.sh", timeout=30)
        if code != 0:
            result.update(reward=None, error=f"sandbox setup failed: {stderr[-200:]}", phase="setup")
            return result

        if is_external:
            # ── External agent mode ──
            # 4a. Extract prompt files from container to OUTPUT_DIR/.prompt/
            prompt_dir = os.path.join(output_dir, ".prompt")
            os.makedirs(prompt_dir, exist_ok=True)
            for fname in ["system.md", "instruction.md", "tool_defs.json", "context.json"]:
                for container_path in [f"/app/{fname}", f"/app/environment/{fname}"]:
                    if docker_cp_from(container_name, container_path, os.path.join(prompt_dir, fname)):
                        break

            # 4b. Get the mapped host port for tool_server
            host_port = docker_port(container_name, 9999)
            if not host_port:
                result.update(reward=None, error="failed to get mapped port for tool_server", phase="setup")
                return result

            # 4c. Run external agent command on the host
            agent_env = os.environ.copy()
            agent_env.update({
                "INSTANCE_ID": instance_id,
                "TOOL_SERVER_URL": f"http://localhost:{host_port}",
                "OUTPUT_DIR": output_dir,
                "SYSTEM_PROMPT_PATH": os.path.join(prompt_dir, "system.md"),
                "INSTRUCTION_PATH": os.path.join(prompt_dir, "instruction.md"),
                "TOOL_DEFS_PATH": os.path.join(prompt_dir, "tool_defs.json"),
                "CONTEXT_PATH": os.path.join(prompt_dir, "context.json"),
            })
            external_cmd = cfg["agent"].get("external_cmd") or cfg["agent"]["cmd"]
            r = subprocess.run(
                external_cmd, shell=True, env=agent_env,
                capture_output=True, text=True,
                timeout=cfg["agent"]["timeout_sec"])
            with open(os.path.join(output_dir, "agent_stdout.txt"), "w") as f:
                f.write(r.stdout)
            with open(os.path.join(output_dir, "agent_stderr.txt"), "w") as f:
                f.write(r.stderr)

            if r.returncode != 0:
                result.update(reward=None, error=f"agent exit {r.returncode}: {r.stderr[-100:]}",
                              phase="agent")
                return result

            # 4d. Copy answer.json from host back into the container
            answer_path = os.path.join(output_dir, "answer.json")
            if not os.path.exists(answer_path):
                result.update(reward=None, error="answer.json not produced", phase="agent")
                return result
            subprocess.run(["docker", "cp", answer_path, f"{container_name}:/app/answer.json"],
                           capture_output=True, text=True, timeout=10)

        else:
            # ── Internal agent mode (existing behavior) ──
            # 4. Run the agent as the agent user (cannot read tests/ or trip.db)
            agent_cmd = cfg["agent"]["cmd"]
            code, stdout, stderr = docker_exec(
                container_name, f"su agent -c '{agent_cmd}'",
                timeout=cfg["agent"]["timeout_sec"])
            with open(os.path.join(output_dir, "agent_stdout.txt"), "w") as f:
                f.write(stdout)
            with open(os.path.join(output_dir, "agent_stderr.txt"), "w") as f:
                f.write(stderr)

            if code != 0:
                result.update(reward=None, error=f"agent exit {code}: {stderr[-100:]}",
                              phase="agent")
                return result

            # Copy agent output to the host
            docker_cp_from(container_name, "/app/answer.json", os.path.join(output_dir, "answer.json"))
            docker_cp_from(container_name, "/app/.tool_calls.jsonl",
                           os.path.join(output_dir, "tool_calls.jsonl"))

            if not os.path.exists(os.path.join(output_dir, "answer.json")):
                result.update(reward=None, error="answer.json not produced", phase="agent")
                return result

        # 5. Run the verifier as root (tests/ and trip.db readable)
        code, stdout, stderr = docker_exec(
            container_name, "cd /app && TESTS_DIR=/app/tests bash tests/test.sh",
            timeout=cfg["verifier"]["timeout_sec"])
        with open(os.path.join(output_dir, "verifier_stdout.txt"), "w") as f:
            f.write(stdout)
        with open(os.path.join(output_dir, "verifier_stderr.txt"), "w") as f:
            f.write(stderr)

        # 6. Extract reward
        code, reward_str, _ = docker_exec(
            container_name,
            "cat /app/reward.txt 2>/dev/null || cat /logs/verifier/reward.txt 2>/dev/null",
            timeout=10)
        if code == 0 and reward_str.strip():
            try:
                reward = float(reward_str.strip())
                result.update(reward=reward, error=None)
            except ValueError:
                result.update(reward=None, error=f"invalid reward: {reward_str.strip()}",
                              phase="verifier")
        else:
            result.update(reward=None, error="reward.txt not found", phase="verifier")

    except subprocess.TimeoutExpired as e:
        result.update(reward=None, error=f"timeout: {e}", phase="timeout")
    except Exception as e:
        result.update(reward=None, error=repr(e)[:200], phase="exception")
    finally:
        docker_rm(container_name)
        result["duration_sec"] = round(time.time() - t0, 1)

    # Save the result of this single task
    with open(os.path.join(output_dir, "result.json"), "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


# ─── Commands ─────────────────────────────────────────────────────────────────

def cmd_run(args):
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)

    print(f"Loading dataset: {cfg['dataset']['repo_id']} config={cfg['dataset']['config']}")
    instances = load_instances(cfg)

    limit = cfg["runner"]["limit"]
    if limit:
        instances = instances[:limit]

    runs = getattr(args, "runs", 1) or 1
    use_run_id = runs > 1

    if args.dry_run:
        print(f"Running {len(instances)} instances × {runs} run(s) (concurrency={cfg['runner']['concurrency']})")
        for inst in instances[:5]:
            image = resolve_image(inst["docker_image"], cfg["docker"]["registry"])
            print(f"  {inst['instance_id']} [{inst.get('domain','')}] → {image}")
        if len(instances) > 5:
            print(f"  ... and {len(instances) - 5} more")
        if use_run_id:
            print(f"  Each instance will be evaluated {runs} times (run_1 .. run_{runs})")
        return

    scores_path = os.path.join(cfg["runner"]["output_dir"], cfg["dataset"]["config"], "scores.jsonl")
    os.makedirs(os.path.dirname(scores_path), exist_ok=True)

    all_results = []
    t0 = time.time()
    concurrency = cfg["runner"]["concurrency"]

    for run_id_val in range(1, runs + 1):
        cur_run_id = run_id_val if use_run_id else None
        if use_run_id:
            print(f"\n{'='*60}")
            print(f"  Run {run_id_val}/{runs}")
            print(f"{'='*60}")

        # Skip done (per run)
        if cfg["runner"]["skip_done"]:
            out_base = os.path.join(cfg["runner"]["output_dir"], cfg["dataset"]["config"])
            todo = []
            for inst in instances:
                if use_run_id:
                    result_path = os.path.join(out_base, inst["instance_id"],
                                               f"run_{run_id_val}", "result.json")
                else:
                    result_path = os.path.join(out_base, inst["instance_id"], "result.json")
                if os.path.exists(result_path):
                    with open(result_path) as f:
                        r = json.load(f)
                    if r.get("reward") is not None:
                        continue
                todo.append(inst)
            skipped = len(instances) - len(todo)
            if skipped:
                print(f"Skipping {skipped} already-completed instances")
            run_instances = todo
        else:
            run_instances = instances

        print(f"Running {len(run_instances)} instances (concurrency={concurrency})")

        if concurrency <= 1:
            for i, inst in enumerate(run_instances):
                result = eval_instance(inst, cfg, run_id=cur_run_id)
                all_results.append(result)
                with open(scores_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                r_str = f"{result['reward']:.3f}" if result["reward"] is not None else f"FAIL({result.get('phase','')}: {result.get('error','')})"
                run_tag = f" run={run_id_val}" if use_run_id else ""
                print(f"[{i+1}/{len(run_instances)}]{run_tag} {inst['instance_id']}: {r_str} ({result.get('duration_sec',0):.0f}s)", flush=True)
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as ex:
                futs = {ex.submit(eval_instance, inst, cfg, run_id=cur_run_id): inst for inst in run_instances}
                for i, fut in enumerate(as_completed(futs)):
                    result = fut.result()
                    all_results.append(result)
                    with open(scores_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    r_str = f"{result['reward']:.3f}" if result["reward"] is not None else f"FAIL({result.get('phase','')})"
                    run_tag = f" run={run_id_val}" if use_run_id else ""
                    print(f"[{i+1}/{len(run_instances)}]{run_tag} {result['instance_id']}: {r_str}", flush=True)

    _print_summary(all_results, cfg, time.time() - t0)


def cmd_status(args):
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)
    out_base = os.path.join(cfg["runner"]["output_dir"], cfg["dataset"]["config"])

    if not os.path.isdir(out_base):
        print(f"No results found at {out_base}")
        return

    scores_path = os.path.join(out_base, "scores.jsonl")
    if not os.path.exists(scores_path):
        print("No scores.jsonl found")
        return

    results = []
    with open(scores_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    _print_summary(results, cfg, None)


def cmd_report(args):
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)
    out_base = os.path.join(cfg["runner"]["output_dir"], cfg["dataset"]["config"])
    scores_path = os.path.join(out_base, "scores.jsonl")

    if not os.path.exists(scores_path):
        print(f"No scores.jsonl at {scores_path}")
        return

    results = []
    with open(scores_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    valid, avg, runs, run_scores, std = _aggregate_pass_at_n(results)
    failed = [r for r in results if r.get("reward") is None]

    summary = {
        "config": cfg["dataset"]["config"],
        "total": len(results),
        "completed": len(valid),
        "failed": len(failed),
        "avg_reward": round(avg, 4),
        "runs": runs,
        "std": round(std, 4),
        "run_scores": run_scores,
        "by_domain": {},
        "failure_breakdown": {},
    }

    # Per-domain
    domain_scores = {}
    for r in valid:
        d = r.get("domain", "unknown")
        domain_scores.setdefault(d, []).append(r["reward"])
    for d, scores in sorted(domain_scores.items()):
        summary["by_domain"][d] = {
            "count": len(scores),
            "avg_reward": round(sum(scores) / len(scores), 4),
            "min": round(min(scores), 4),
            "max": round(max(scores), 4),
        }

    # Failure breakdown
    for r in failed:
        phase = r.get("phase", "unknown")
        summary["failure_breakdown"][phase] = summary["failure_breakdown"].get(phase, 0) + 1

    summary_path = os.path.join(out_base, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved to {summary_path}")


def cmd_submit(args):
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args)
    dataset_config = cfg["dataset"]["config"]
    out_base = os.path.join(cfg["runner"]["output_dir"], dataset_config)

    scores_path = os.path.join(out_base, "scores.jsonl")
    if not os.path.exists(scores_path):
        print(f"ERROR: no scores.jsonl at {scores_path}", file=sys.stderr)
        print("Run evaluation first: flyai-bench run", file=sys.stderr)
        sys.exit(1)

    results = []
    with open(scores_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))

    if not any(r.get("reward") is not None for r in results):
        print("ERROR: no valid scores found", file=sys.stderr)
        sys.exit(1)

    model_name = args.model or cfg["agent"].get("llm_model", "unknown")
    agent_type = args.agent_type or "mini-swe-agent"
    date_str = time.strftime("%Y%m%d")
    slug = f"{date_str}_{model_name}_{agent_type}".replace("/", "_").replace(" ", "_")
    submit_dir = os.path.join(args.submit_dir or "experiments/evaluation", dataset_config, slug)
    os.makedirs(submit_dir, exist_ok=True)

    import shutil
    shutil.copy2(scores_path, os.path.join(submit_dir, "scores.jsonl"))

    summary_path = os.path.join(out_base, "summary.json")
    if os.path.exists(summary_path):
        shutil.copy2(summary_path, os.path.join(submit_dir, "summary.json"))
    else:
        summary = _build_summary(results, dataset_config, model_name, agent_type)
        with open(os.path.join(submit_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    valid, avg_reward, runs, run_scores, std = _aggregate_pass_at_n(results)
    avg_reward = round(avg_reward, 4)
    failed = [r for r in results if r.get("reward") is None]
    eval_section = {
        "benchmark": "ecommerce_last_exam",
        "config": dataset_config,
        "total_instances": len(results),
        "completed": len(valid),
        "failed": len(failed),
        "avg_reward": avg_reward,
        "date": time.strftime("%Y-%m-%d"),
    }
    if runs > 1:
        eval_section["runs"] = runs
        eval_section["std"] = round(std, 4)
    metadata = {
        "model": {
            "name": model_name,
            "provider": args.provider or "",
            "api_params": {
                "temperature": 0.0,
                "max_tokens": 8192,
            },
        },
        "agent": {
            "type": agent_type,
            "version": "1.0",
            "max_iterations": 30,
            "tool_calling": "terminal",
        },
        "evaluation": eval_section,
        "notes": args.notes or "",
    }
    with open(os.path.join(submit_dir, "metadata.yaml"), "w", encoding="utf-8") as f:
        if yaml:
            yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        else:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"Submission packaged at: {submit_dir}/")
    print(f"  metadata.yaml  — edit model/agent details before submitting")
    print(f"  scores.jsonl   — {len(results)} records ({len(valid)} valid, {len(failed)} failed)")
    print(f"  summary.json   — avg_reward={avg_reward:.4f}" + (f"  runs={runs}  std=±{std:.4f}" if runs > 1 else ""))
    print()
    print("Next steps:")
    print(f"  1. Review and edit {submit_dir}/metadata.yaml")
    print(f"  2. git add {submit_dir} && git commit")
    print(f"  3. Open a Pull Request to flyai-bench main branch")


def _build_summary(results, dataset_config, model_name, agent_type):
    valid, avg, runs, run_scores, std = _aggregate_pass_at_n(results)
    failed = [r for r in results if r.get("reward") is None]
    summary = {
        "config": dataset_config,
        "model": model_name,
        "agent": agent_type,
        "total": len(results),
        "completed": len(valid),
        "failed": len(failed),
        "avg_reward": round(avg, 4),
        "runs": runs,
        "std": round(std, 4),
        "run_scores": run_scores,
        "by_domain": {},
        "failure_breakdown": {},
    }
    domain_scores = {}
    for r in valid:
        d = r.get("domain", "unknown")
        domain_scores.setdefault(d, []).append(r["reward"])
    for d, scores in sorted(domain_scores.items()):
        summary["by_domain"][d] = {
            "count": len(scores),
            "avg_reward": round(sum(scores) / len(scores), 4),
            "min": round(min(scores), 4),
            "max": round(max(scores), 4),
        }
    for r in failed:
        phase = r.get("phase", "unknown")
        summary["failure_breakdown"][phase] = summary["failure_breakdown"].get(phase, 0) + 1
    return summary


def _aggregate_pass_at_n(results):
    """Aggregate grouped by instance_id, return (instance_means, runs, run_scores, std)."""
    valid = [r for r in results if r.get("reward") is not None]
    has_run_id = any(r.get("run_id") is not None for r in valid)

    if not has_run_id:
        avg = sum(r["reward"] for r in valid) / len(valid) if valid else 0
        return valid, avg, 1, [], 0.0

    from collections import defaultdict
    by_instance = defaultdict(list)
    by_run = defaultdict(list)
    for r in valid:
        by_instance[r["instance_id"]].append(r["reward"])
        by_run[r["run_id"]].append(r["reward"])

    instance_means = [sum(rs) / len(rs) for rs in by_instance.values()]
    avg = sum(instance_means) / len(instance_means) if instance_means else 0

    run_ids = sorted(by_run.keys())
    runs = len(run_ids)
    run_scores = [round(sum(by_run[rid]) / len(by_run[rid]), 4) for rid in run_ids]

    std = 0.0
    if runs > 1:
        mean_of_runs = sum(run_scores) / len(run_scores)
        variance = sum((s - mean_of_runs) ** 2 for s in run_scores) / (len(run_scores) - 1)
        std = variance ** 0.5

    return valid, avg, runs, run_scores, std


def _print_summary(results, cfg, elapsed):
    valid, avg, runs, run_scores, std = _aggregate_pass_at_n(results)
    failed_count = len(results) - len(valid)

    print(f"\n{'='*60}")
    print(f"  Config: {cfg['dataset']['config']}")
    print(f"  Completed: {len(valid)}/{len(results)}" + (f" (failed: {failed_count})" if failed_count else ""))
    print(f"  Avg Reward: {avg:.4f}")
    if runs > 1:
        print(f"  Runs: {runs}  Std: ±{std:.4f}")
        print(f"  Per-run scores: {', '.join(f'{s:.4f}' for s in run_scores)}")
    if elapsed:
        print(f"  Elapsed: {elapsed/60:.1f} min")

    domain_scores = {}
    for r in valid:
        d = r.get("domain", "unknown")
        domain_scores.setdefault(d, []).append(r["reward"])
    if domain_scores:
        print(f"  By domain:")
        for d, scores in sorted(domain_scores.items()):
            print(f"    {d}: {sum(scores)/len(scores):.4f} (n={len(scores)})")
    print(f"{'='*60}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="flyai-bench",
        description="E-Commerce Last Exam Benchmark Evaluation CLI",
    )
    default_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_config.yaml")
    parser.add_argument("--config", default=default_config,
                        help="Config file path (default: built-in eval_config.yaml)")

    sub = parser.add_subparsers(dest="command")

    # run
    p_run = sub.add_parser("run", help="Run evaluation")
    p_run.add_argument("--dataset-config", choices=["travel", "e_commerce"])
    p_run.add_argument("--limit", type=int)
    p_run.add_argument("--concurrency", type=int)
    p_run.add_argument("--output-dir")
    p_run.add_argument("--agent-cmd",
                        help="Agent command (internal: runs inside container; external: runs on host)")
    p_run.add_argument("--agent-mode", choices=["internal", "external"],
                        help="Agent execution mode (default: internal)")
    p_run.add_argument("--registry")
    p_run.add_argument("--runs", type=int, default=1,
                        help="Number of independent runs (pass@N, default: 1)")
    p_run.add_argument("--dry-run", action="store_true")

    # status
    p_status = sub.add_parser("status", help="Check evaluation progress")
    p_status.add_argument("--dataset-config", choices=["travel", "e_commerce"])
    p_status.add_argument("--output-dir")

    # report
    p_report = sub.add_parser("report", help="Generate an evaluation report")
    p_report.add_argument("--dataset-config", choices=["travel", "e_commerce"])
    p_report.add_argument("--output-dir")

    # submit
    p_submit = sub.add_parser("submit", help="Package evaluation results for leaderboard submission")
    p_submit.add_argument("--dataset-config", choices=["travel", "e_commerce"])
    p_submit.add_argument("--output-dir")
    p_submit.add_argument("--submit-dir", default="experiments/evaluation",
                          help="Output directory for the submission package (default: experiments/evaluation/)")
    p_submit.add_argument("--model", help="Model name (e.g. deepseek-v4-flash)")
    p_submit.add_argument("--provider", help="Model provider (e.g. deepseek)")
    p_submit.add_argument("--agent-type", default="mini-swe-agent",
                          help="Agent type (default: mini-swe-agent)")
    p_submit.add_argument("--notes", help="Submission notes")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "run":
        cmd_run(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "report":
        cmd_report(args)
    elif args.command == "submit":
        cmd_submit(args)


if __name__ == "__main__":
    main()
