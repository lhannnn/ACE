#!/usr/bin/env python3
import argparse
import glob
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from common import (
    ACE_FULL_RESULTS_ROOT,
    ACE_QUICK_RESULTS_ROOT,
    ACE_ROOT,
    DATASET_TO_ACE_TASK,
    MANIFEST_PATH,
)


REQUIRED_RESULT_FILES = [
    "final_results.json",
    "test_results.json",
    "train_results.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ACE online for aligned fast20 datasets.")
    parser.add_argument("--manifest_path", type=Path, default=MANIFEST_PATH)
    parser.add_argument(
        "--config_path",
        type=Path,
        default=ACE_ROOT / "eval" / "abstention" / "data" / "abstention_config_fast20_aligned.json",
    )
    parser.add_argument(
        "--profile",
        type=str,
        choices=["quick", "full"],
        required=True,
        help="quick: max_num_rounds=1; full: max_num_rounds=3",
    )
    parser.add_argument("--results_root", type=Path, default=None)
    parser.add_argument("--registry_path", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only_dataset", type=str, default=None)
    parser.add_argument("--continue_on_error", action="store_true")
    return parser.parse_args()


def default_results_root(profile: str) -> Path:
    if profile == "quick":
        return ACE_QUICK_RESULTS_ROOT
    return ACE_FULL_RESULTS_ROOT


def default_registry_path(profile: str) -> Path:
    return Path(__file__).resolve().parent / f"run_registry_ace_{profile}.json"


def load_registry(path: Path) -> dict[str, Any]:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"runs": {}}


def save_registry(path: Path, registry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
        f.write("\n")


def normalize_metrics(raw: dict[str, Any] | None) -> dict[str, Any]:
    if not raw:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
            "total": 0,
            "evaluated_total": 0,
            "indeterminate": 0,
        }
    total = int(raw.get("total", 0))
    indeterminate = int(raw.get("indeterminate", 0))
    evaluated_total = int(raw.get("evaluated_total", total - indeterminate))
    return {
        "precision": float(raw.get("precision", 0.0)),
        "recall": float(raw.get("recall", 0.0)),
        "f1": float(raw.get("f1", 0.0)),
        "tp": int(raw.get("tp", 0)),
        "fp": int(raw.get("fp", 0)),
        "fn": int(raw.get("fn", 0)),
        "tn": int(raw.get("tn", 0)),
        "total": total,
        "evaluated_total": evaluated_total,
        "indeterminate": indeterminate,
    }


def find_latest_ace_run_dir(results_root: Path, task_name: str) -> Path:
    pattern = str(results_root / f"ace_run_*_{task_name}_online")
    candidates = [Path(p) for p in glob.glob(pattern)]
    if not candidates:
        raise FileNotFoundError(f"No ACE run directory matches pattern: {pattern}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def build_command(
    task_name: str,
    config_path: Path,
    results_root: Path,
    max_num_rounds: int,
) -> list[str]:
    command = (
        "set -a; "
        "source .env; "
        "set +a; "
        "python -m eval.abstention.run "
        f"--task_name {shlex.quote(task_name)} "
        "--mode online "
        "--api_provider together "
        "--judge_api_provider together "
        "--judge_model meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo "
        "--strict_judge "
        f"--config_path {shlex.quote(str(config_path))} "
        f"--save_path {shlex.quote(str(results_root))} "
        f"--max_num_rounds {max_num_rounds}"
    )
    return ["/bin/zsh", "-lc", command]


def main() -> int:
    args = parse_args()
    results_root = args.results_root or default_results_root(args.profile)
    registry_path = args.registry_path or default_registry_path(args.profile)
    max_num_rounds = 1 if args.profile == "quick" else 3

    with open(args.manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    dataset_order = list(manifest["dataset_order"])
    if args.only_dataset:
        if args.only_dataset not in dataset_order:
            raise ValueError(f"{args.only_dataset} is not in manifest dataset_order")
        dataset_order = [args.only_dataset]

    registry = load_registry(registry_path)
    registry["updated_at"] = datetime.now().isoformat(timespec="seconds")
    registry["profile"] = args.profile
    registry["manifest_path"] = str(args.manifest_path)
    registry["config_path"] = str(args.config_path)
    registry["results_root"] = str(results_root)
    registry.setdefault("runs", {})

    failures = 0
    for dataset_name in dataset_order:
        if dataset_name not in DATASET_TO_ACE_TASK:
            raise KeyError(f"Missing ACE task mapping for dataset: {dataset_name}")
        task_name = DATASET_TO_ACE_TASK[dataset_name]

        previous = registry["runs"].get(dataset_name)
        if args.resume and previous and previous.get("status") == "success":
            print(f"[SKIP] {dataset_name} ({task_name}) already successful")
            continue

        cmd = build_command(
            task_name=task_name,
            config_path=args.config_path,
            results_root=results_root,
            max_num_rounds=max_num_rounds,
        )
        command_str = " ".join(shlex.quote(x) for x in cmd)
        print(f"\n[RUN] {dataset_name} -> {task_name} ({args.profile})")
        print(f"[CMD] {command_str}")

        started_at = datetime.now().isoformat(timespec="seconds")
        result = subprocess.run(cmd, cwd=ACE_ROOT)
        finished_at = datetime.now().isoformat(timespec="seconds")

        run_entry: dict[str, Any] = {
            "dataset": dataset_name,
            "task_name": task_name,
            "profile": args.profile,
            "status": "failed",
            "started_at": started_at,
            "finished_at": finished_at,
            "return_code": result.returncode,
            "command": command_str,
        }

        if result.returncode == 0:
            try:
                run_dir = find_latest_ace_run_dir(results_root, task_name)
                run_entry["run_dir"] = str(run_dir)

                missing_files = [
                    name for name in REQUIRED_RESULT_FILES if not (run_dir / name).exists()
                ]
                run_entry["missing_result_files"] = missing_files
                if missing_files:
                    run_entry["status"] = "missing_files"
                else:
                    with open(run_dir / "final_results.json", "r", encoding="utf-8") as f:
                        final_payload = json.load(f)

                    initial_metrics = normalize_metrics(final_payload.get("initial_test_results"))
                    online_metrics = normalize_metrics(final_payload.get("online_test_results"))
                    run_entry["initial_metrics"] = initial_metrics
                    run_entry["online_metrics"] = online_metrics

                    run_entry["initial_total_consistent"] = (
                        initial_metrics["evaluated_total"] + initial_metrics["indeterminate"]
                        == initial_metrics["total"]
                    )
                    run_entry["online_total_consistent"] = (
                        online_metrics["evaluated_total"] + online_metrics["indeterminate"]
                        == online_metrics["total"]
                    )
                    run_entry["status"] = "success"
            except Exception as e:  # noqa: BLE001
                run_entry["status"] = "postprocess_error"
                run_entry["error"] = f"{type(e).__name__}: {e}"

        registry["runs"][dataset_name] = run_entry
        save_registry(registry_path, registry)

        if run_entry["status"] != "success":
            failures += 1
            print(f"[FAIL] {dataset_name}: status={run_entry['status']}")
            if not args.continue_on_error:
                print("[STOP] Exiting due to failure. Use --continue_on_error to continue.")
                return 1
        else:
            m = run_entry["online_metrics"]
            print(
                f"[OK] {dataset_name}: "
                f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
                f"(total={m['total']}, ind={m['indeterminate']})"
            )

    print(f"\n[DONE] Completed ACE {args.profile} run loop. failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
