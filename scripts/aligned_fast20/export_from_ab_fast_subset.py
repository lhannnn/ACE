#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from hydra.utils import instantiate
from omegaconf import OmegaConf

from common import (
    AB_ROOT,
    ALIGNED_DATA_DIR,
    DATASET_TO_ACE_TASK,
    EXCLUDED_DATASETS,
    FAST_INDICES_PATH,
    MANIFEST_PATH,
)


DEFAULT_SCENARIO_BY_DATASET = {
    "ALCUNADataset": "underspecified_context",
    "BBQDataset": "underspecified_context",
    "BigBenchDisambiguateDataset": "underspecified_context",
    "CoCoNotDataset": "unsupported_stale",
    "FalseQADataset": "false_premise",
    "FreshQADataset": "unsupported_stale",
    "GSM8K": "underspecified_context",
    "KUQDataset": "unsupported_stale",
    "MediQDataset": "underspecified_context",
    "MMLUMath": "underspecified_context",
    "MoralChoiceDataset": "unsupported_stale",
    "MusiqueDataset": "underspecified_context",
    "NQDataset": "underspecified_context",
    "QAQADataset": "false_premise",
    "QASPERDataset": "underspecified_context",
    "SelfAwareDataset": "answer_unknown",
    "SituatedQAGeoDataset": "underspecified_context",
    "Squad2Dataset": "underspecified_context",
    "UMWP": "underspecified_context",
    "WorldSenseDataset": "underspecified_context",
}

# Allow hydra.instantiate() to resolve recipe.* modules from AbstentionBench.
if str(AB_ROOT) not in sys.path:
    sys.path.insert(0, str(AB_ROOT))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def flatten_strings(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        out: list[str] = []
        for k, v in value.items():
            out.append(str(k))
            out.extend(flatten_strings(v))
        return out
    if isinstance(value, (list, tuple, set)):
        out: list[str] = []
        for item in value:
            out.extend(flatten_strings(item))
        return out
    return [str(value)]


def to_json_compatible(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): to_json_compatible(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_compatible(v) for v in value]
    if isinstance(value, set):
        return [to_json_compatible(v) for v in sorted(value, key=str)]
    return str(value)


def map_scenario(dataset_name: str, metadata: dict[str, Any]) -> str:
    text = " ".join(x.lower() for x in flatten_strings(metadata))
    text = f"{dataset_name.lower()} {text}"

    if any(token in text for token in ("false premise", "false_assumption", "false assumption")):
        return "false_premise"
    if any(token in text for token in ("answer unknown", "future_unknown", "unsolved", "unknown answer")):
        return "answer_unknown"
    if any(token in text for token in ("ambiguous", "underspecified", "disambiguate")):
        return "underspecified_context"
    if any(token in text for token in ("subjective", "stale", "safety", "unsupported")):
        return "unsupported_stale"
    return DEFAULT_SCENARIO_BY_DATASET.get(dataset_name, "unsupported_stale")


def discover_dataset_config_paths(dataset_config_dir: Path) -> dict[str, Path]:
    config_by_dataset_name: dict[str, Path] = {}
    for yaml_path in sorted(dataset_config_dir.glob("*.yaml")):
        cfg = OmegaConf.load(yaml_path)
        dataset_name = cfg.get("dataset_name")
        if dataset_name:
            config_by_dataset_name[str(dataset_name)] = yaml_path
    return config_by_dataset_name


@contextmanager
def pushd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def instantiate_dataset(config_path: Path, ab_data_dir: Path):
    cfg = OmegaConf.load(config_path)
    merged_cfg = OmegaConf.merge(cfg, OmegaConf.create({"data_dir": str(ab_data_dir)}))
    OmegaConf.resolve(merged_cfg)

    with pushd(AB_ROOT):
        dataset = instantiate(merged_cfg.datamodule)
    return dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export AB fast-subset datasets to ACE-aligned JSONL files."
    )
    parser.add_argument(
        "--fast_indices_path",
        type=Path,
        default=FAST_INDICES_PATH,
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=ALIGNED_DATA_DIR,
    )
    parser.add_argument(
        "--manifest_path",
        type=Path,
        default=MANIFEST_PATH,
    )
    parser.add_argument(
        "--only_dataset",
        type=str,
        default=None,
        help="Export only one dataset key from fast-subset-indices.json.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing dataset JSONL files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.fast_indices_path, "r", encoding="utf-8") as f:
        fast_indices = json.load(f)

    dataset_order = [
        name for name in fast_indices.keys()
        if name not in EXCLUDED_DATASETS
    ]
    if args.only_dataset:
        if args.only_dataset not in dataset_order:
            raise ValueError(f"{args.only_dataset} is not in available fast-subset datasets.")
        dataset_order = [args.only_dataset]

    config_paths = discover_dataset_config_paths(AB_ROOT / "configs" / "dataset")

    missing_configs = [name for name in dataset_order if name not in config_paths]
    if missing_configs:
        raise RuntimeError(f"Missing AB dataset configs for: {missing_configs}")

    missing_task_mapping = [name for name in dataset_order if name not in DATASET_TO_ACE_TASK]
    if missing_task_mapping:
        raise RuntimeError(f"Missing ACE task mapping for datasets: {missing_task_mapping}")

    manifest: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "fast_indices_path": str(args.fast_indices_path),
        "output_dir": str(args.output_dir),
        "excluded_datasets": sorted(EXCLUDED_DATASETS),
        "dataset_order": dataset_order,
        "datasets": {},
    }

    ab_data_dir = AB_ROOT / "data"

    for dataset_name in dataset_order:
        indices = list(fast_indices[dataset_name])
        config_path = config_paths[dataset_name]
        output_path = args.output_dir / f"{dataset_name}.jsonl"

        if output_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing file: {output_path} (pass --overwrite)."
            )

        print(f"[EXPORT] {dataset_name} ({len(indices)} indices) from {config_path.name}")
        dataset = instantiate_dataset(config_path, ab_data_dir)

        rows: list[dict[str, Any]] = []
        source_questions: list[str] = []
        scenario_counter: Counter[str] = Counter()
        for source_idx in indices:
            if source_idx >= len(dataset):
                raise IndexError(
                    f"{dataset_name} index {source_idx} out of range (dataset size={len(dataset)})"
                )
            prompt = dataset[source_idx]
            if prompt.should_abstain is None:
                raise ValueError(
                    f"{dataset_name} index {source_idx} has should_abstain=None; "
                    "cannot export for abstention evaluation."
                )

            metadata = to_json_compatible(dict(prompt.metadata or {}))
            scenario = map_scenario(dataset_name, metadata)
            scenario_counter[scenario] += 1

            source_questions.append(prompt.question)
            rows.append(
                {
                    "question": prompt.question,
                    "reference_answers": to_json_compatible(prompt.reference_answers),
                    "should_abstain": bool(prompt.should_abstain),
                    "scenario": scenario,
                    "metadata": {
                        **metadata,
                        "aligned_fast20": {
                            "source_dataset_name": dataset_name,
                            "source_index": source_idx,
                            "source_config": config_path.name,
                        },
                    },
                }
            )

        with open(output_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        # Round-trip validation: exported questions should match source ordering.
        exported_questions: list[str] = []
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                exported_questions.append(json.loads(line)["question"])

        question_match_count = sum(
            1 for src_q, exp_q in zip(source_questions, exported_questions) if src_q == exp_q
        )

        manifest["datasets"][dataset_name] = {
            "ace_task_name": DATASET_TO_ACE_TASK[dataset_name],
            "source_config": str(config_path),
            "output_file": str(output_path),
            "record_count": len(rows),
            "indices_count": len(indices),
            "indices_sha256": sha256_text(json.dumps(indices, ensure_ascii=False)),
            "question_sha256": sha256_text("\n".join(source_questions)),
            "file_sha256": sha256_file(output_path),
            "question_match_count": question_match_count,
            "question_mismatch_count": len(source_questions) - question_match_count,
            "scenario_counts": dict(sorted(scenario_counter.items())),
        }

        print(
            f"[OK] {dataset_name}: wrote {len(rows)} rows, "
            f"question_match={question_match_count}/{len(source_questions)}"
        )

    manifest["dataset_count"] = len(manifest["datasets"])
    manifest["total_records"] = sum(
        item["record_count"] for item in manifest["datasets"].values()
    )

    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"\n[DONE] Export complete. Manifest: {args.manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
