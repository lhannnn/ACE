#!/usr/bin/env python3
import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from common import AB_RESULTS_ROOT, AB_ROOT, ACE_ROOT, MANIFEST_PATH


REQUIRED_RESULT_FILES = [
    "InferencePipeline.json",
    "DirectAbstention.json",
    "LLMJudgeAbstentionDetector.json",
    "GroundTruthAbstentionEvaluator.json",
    "config.json",
]


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


def compute_ab_metrics(evaluator_path: Path) -> dict[str, Any]:
    with open(evaluator_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    responses = payload.get("responses", [])

    tp = fp = fn = tn = 0
    indeterminate = 0
    for row in responses:
        should_abstain = row["prompt"].get("should_abstain")
        is_abstention = row.get("is_abstention")
        if is_abstention is None:
            indeterminate += 1
            continue
        if should_abstain and is_abstention:
            tp += 1
        elif should_abstain and not is_abstention:
            fn += 1
        elif (not should_abstain) and is_abstention:
            fp += 1
        else:
            tn += 1

    evaluated_total = tp + fp + fn + tn
    total = evaluated_total + indeterminate
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "total": total,
        "evaluated_total": evaluated_total,
        "indeterminate": indeterminate,
    }


def latest_run_dir(dataset_name: str, results_root: Path) -> Path:
    run_base = results_root / f"{dataset_name}_TogetherAIAPI"
    if not run_base.exists():
        raise FileNotFoundError(f"Missing run base directory: {run_base}")
    run_dirs = [p for p in run_base.iterdir() if p.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found under: {run_base}")
    return max(run_dirs, key=lambda p: p.stat().st_mtime)


def build_command(dataset_name: str, data_path: Path) -> list[str]:
    command = (
        "set -a; "
        f"source {shlex.quote(str(ACE_ROOT / '.env'))}; "
        "set +a; "
        "python main.py -m "
        "mode=local "
        "dataset=ace_jsonl "
        "model=together_ai "
        "abstention_detector=llm_judge_togetherai "
        "run_single_job_for_inference_and_judge=True "
        "common_dir=$(pwd) "
        "sweep_folder=aligned_fast20 "
        f"dataset_name={shlex.quote(dataset_name)} "
        f"datamodule.data_path={shlex.quote(str(data_path))} "
        "dataset_indices_path=null "
        "module.temperature=0.0"
    )
    return ["/bin/zsh", "-lc", command]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AB on aligned fast20 JSONL datasets.")
    parser.add_argument("--manifest_path", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--results_root", type=Path, default=AB_RESULTS_ROOT)
    parser.add_argument(
        "--registry_path",
        type=Path,
        default=Path(__file__).resolve().parent / "run_registry_ab.json",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only_dataset", type=str, default=None)
    parser.add_argument("--continue_on_error", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with open(args.manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    dataset_order = list(manifest["dataset_order"])
    if args.only_dataset:
        if args.only_dataset not in dataset_order:
            raise ValueError(f"{args.only_dataset} is not in manifest dataset_order")
        dataset_order = [args.only_dataset]

    registry = load_registry(args.registry_path)
    registry["updated_at"] = datetime.now().isoformat(timespec="seconds")
    registry["manifest_path"] = str(args.manifest_path)
    registry["results_root"] = str(args.results_root)
    registry.setdefault("runs", {})

    failures = 0
    for dataset_name in dataset_order:
        previous = registry["runs"].get(dataset_name)
        if args.resume and previous and previous.get("status") == "success":
            print(f"[SKIP] {dataset_name} (already successful)")
            continue

        dataset_info = manifest["datasets"][dataset_name]
        data_path = Path(dataset_info["output_file"])
        if not data_path.exists():
            raise FileNotFoundError(f"Aligned JSONL not found: {data_path}")

        cmd = build_command(dataset_name, data_path)
        command_str = " ".join(shlex.quote(x) for x in cmd)
        print(f"\n[RUN] {dataset_name}")
        print(f"[CMD] {command_str}")

        started_at = datetime.now().isoformat(timespec="seconds")
        result = subprocess.run(cmd, cwd=AB_ROOT)
        finished_at = datetime.now().isoformat(timespec="seconds")

        run_entry: dict[str, Any] = {
            "dataset": dataset_name,
            "status": "failed",
            "started_at": started_at,
            "finished_at": finished_at,
            "return_code": result.returncode,
            "command": command_str,
            "aligned_data_path": str(data_path),
        }

        # Capture latest run directory even when the job fails, so logs are easy to inspect.
        try:
            run_entry["run_dir"] = str(latest_run_dir(dataset_name, args.results_root))
        except Exception:  # noqa: BLE001
            pass

        if result.returncode == 0:
            try:
                run_dir = Path(run_entry["run_dir"])
                run_entry["run_dir"] = str(run_dir)

                missing_files = [
                    name for name in REQUIRED_RESULT_FILES if not (run_dir / name).exists()
                ]
                run_entry["missing_result_files"] = missing_files

                if not missing_files:
                    metrics = compute_ab_metrics(run_dir / "GroundTruthAbstentionEvaluator.json")
                    run_entry["metrics"] = metrics
                    with open(run_dir / "config.json", "r", encoding="utf-8") as f:
                        config_payload = json.load(f)
                    run_entry["module_temperature"] = (
                        config_payload.get("module", {}).get("temperature")
                    )
                    run_entry["temperature_ok"] = run_entry["module_temperature"] == 0.0
                    run_entry["status"] = "success"
                else:
                    run_entry["status"] = "missing_files"
            except Exception as e:  # noqa: BLE001
                run_entry["status"] = "postprocess_error"
                run_entry["error"] = f"{type(e).__name__}: {e}"
        else:
            run_entry["status"] = "failed"

        registry["runs"][dataset_name] = run_entry
        save_registry(args.registry_path, registry)

        if run_entry["status"] != "success":
            failures += 1
            print(f"[FAIL] {dataset_name}: status={run_entry['status']}")
            if not args.continue_on_error:
                print("[STOP] Exiting due to failure. Use --continue_on_error to continue.")
                return 1
        else:
            m = run_entry["metrics"]
            print(
                f"[OK] {dataset_name}: "
                f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
                f"(total={m['total']}, ind={m['indeterminate']})"
            )

    print(f"\n[DONE] Completed AB run loop. failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
