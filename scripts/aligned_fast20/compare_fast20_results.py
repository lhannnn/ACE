#!/usr/bin/env python3
import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from common import REPORT_DIR


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def empty_metrics() -> dict[str, Any]:
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


def normalize_metrics(metrics: dict[str, Any] | None) -> dict[str, Any]:
    if not metrics:
        return empty_metrics()
    out = empty_metrics()
    out.update({k: metrics.get(k, out[k]) for k in out})
    out["precision"] = float(out["precision"])
    out["recall"] = float(out["recall"])
    out["f1"] = float(out["f1"])
    out["tp"] = int(out["tp"])
    out["fp"] = int(out["fp"])
    out["fn"] = int(out["fn"])
    out["tn"] = int(out["tn"])
    out["total"] = int(out["total"])
    out["evaluated_total"] = int(out["evaluated_total"])
    out["indeterminate"] = int(out["indeterminate"])
    return out


def format_metric_cell(metrics: dict[str, Any] | None, status: str | None = None) -> str:
    if not metrics:
        return f"NA ({status or 'missing'})"
    m = normalize_metrics(metrics)
    return (
        f"{m['precision']:.3f}/{m['recall']:.3f}/{m['f1']:.3f} | "
        f"{m['tp']}/{m['fp']}/{m['fn']}/{m['tn']} | "
        f"{m['total']}/{m['evaluated_total']}/{m['indeterminate']}"
    )


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Compare AB and ACE fast20 aligned runs.")
    parser.add_argument("--manifest_path", type=Path, required=True)
    parser.add_argument("--ab_registry", type=Path, default=base / "run_registry_ab.json")
    parser.add_argument("--ace_quick_registry", type=Path, default=base / "run_registry_ace_quick.json")
    parser.add_argument("--ace_full_registry", type=Path, default=base / "run_registry_ace_full.json")
    parser.add_argument("--report_dir", type=Path, default=REPORT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_json(args.manifest_path)
    ab_registry = load_json(args.ab_registry) if args.ab_registry.exists() else {"runs": {}}
    ace_quick_registry = (
        load_json(args.ace_quick_registry) if args.ace_quick_registry.exists() else {"runs": {}}
    )
    ace_full_registry = (
        load_json(args.ace_full_registry) if args.ace_full_registry.exists() else {"runs": {}}
    )

    dataset_order = list(manifest["dataset_order"])
    rows: list[dict[str, Any]] = []

    for dataset in dataset_order:
        ab_entry = ab_registry.get("runs", {}).get(dataset, {})
        ace_q_entry = ace_quick_registry.get("runs", {}).get(dataset, {})
        ace_f_entry = ace_full_registry.get("runs", {}).get(dataset, {})

        ab_metrics = ab_entry.get("metrics") if ab_entry.get("status") == "success" else None

        quick_initial = (
            ace_q_entry.get("initial_metrics") if ace_q_entry.get("status") == "success" else None
        )
        full_initial = (
            ace_f_entry.get("initial_metrics") if ace_f_entry.get("status") == "success" else None
        )
        # Prefer initial metrics from full profile when available, else quick.
        ace_initial = full_initial or quick_initial
        ace_quick_final = (
            ace_q_entry.get("online_metrics") if ace_q_entry.get("status") == "success" else None
        )
        ace_full_final = (
            ace_f_entry.get("online_metrics") if ace_f_entry.get("status") == "success" else None
        )

        notes: list[str] = []
        if dataset == "AveritecDataset":
            notes.append("Excluded by design.")
        if quick_initial and full_initial and normalize_metrics(quick_initial) != normalize_metrics(full_initial):
            notes.append("ACE initial differs between quick/full runs.")
        if ab_entry.get("status") not in (None, "success"):
            notes.append(f"AB status={ab_entry.get('status')}")
        if ace_q_entry.get("status") not in (None, "success"):
            notes.append(f"ACE quick status={ace_q_entry.get('status')}")
        if ace_f_entry.get("status") not in (None, "success"):
            notes.append(f"ACE full status={ace_f_entry.get('status')}")

        rows.append(
            {
                "dataset": dataset,
                "ab": normalize_metrics(ab_metrics) if ab_metrics else None,
                "ace_initial": normalize_metrics(ace_initial) if ace_initial else None,
                "ace_quick_final": normalize_metrics(ace_quick_final) if ace_quick_final else None,
                "ace_full_final": normalize_metrics(ace_full_final) if ace_full_final else None,
                "ab_status": ab_entry.get("status"),
                "ace_quick_status": ace_q_entry.get("status"),
                "ace_full_status": ace_f_entry.get("status"),
                "notes": notes,
            }
        )

    generated_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.report_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.report_dir / f"aligned_fast20_compare_{generated_at}.json"
    md_path = args.report_dir / f"aligned_fast20_compare_{generated_at}.md"

    output_payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "excluded_dataset_note": "AveritecDataset is excluded in this run by design.",
        "rows": rows,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)
        f.write("\n")

    lines = []
    lines.append("# Aligned Fast20 Comparison (AB vs ACE)")
    lines.append("")
    lines.append("- Scope: fast-subset datasets excluding `AveritecDataset`.")
    lines.append("- Metric format in table cells: `P/R/F1 | TP/FP/FN/TN | total/evaluated_total/indeterminate`.")
    lines.append("")
    lines.append(
        "| Dataset | AB | ACE initial | ACE final (quick) | ACE final (full) | Notes |"
    )
    lines.append("|---|---|---|---|---|---|")
    for row in rows:
        notes = "; ".join(row["notes"]) if row["notes"] else ""
        lines.append(
            "| {dataset} | {ab} | {initial} | {quick} | {full} | {notes} |".format(
                dataset=row["dataset"],
                ab=format_metric_cell(row["ab"], row["ab_status"]),
                initial=format_metric_cell(
                    row["ace_initial"],
                    row["ace_full_status"] or row["ace_quick_status"],
                ),
                quick=format_metric_cell(row["ace_quick_final"], row["ace_quick_status"]),
                full=format_metric_cell(row["ace_full_final"], row["ace_full_status"]),
                notes=notes,
            )
        )
    lines.append("")
    lines.append("Averitec was intentionally excluded from this batch.")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")

    print(f"[DONE] Wrote JSON report: {json_path}")
    print(f"[DONE] Wrote Markdown report: {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
