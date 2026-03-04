#!/usr/bin/env python3
"""
Post-processing evaluation script that re-evaluates ACE results
using AbstentionBench's LLM Judge for abstention detection.

Usage:
  TOGETHER_API_KEY=xxx python -m eval.abstention.evaluate_with_judge \
      --results_dir results/ace_run_XXXXX_coconot_online_online \
      --data_path eval/abstention/data/coconot_all.jsonl \
      --api_provider together \
      --judge_model meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo
"""
import os
import re
import json
import glob
import time
import argparse
import openai
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from .data_processor import DataProcessor, load_data, LLM_JUDGE_ABSTENTION_PROMPT


def parse_args():
    parser = argparse.ArgumentParser(
        description="Re-evaluate ACE results with AbstentionBench LLM Judge"
    )
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Path to ACE run results directory")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the JSONL data file used for the run")
    parser.add_argument("--api_provider", type=str, default="together",
                        choices=["together", "sambanova", "openai"])
    parser.add_argument("--judge_model", type=str,
                        default="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
    parser.add_argument("--max_workers", type=int, default=10)
    parser.add_argument("--eval_type", type=str, default="initial",
                        choices=["initial", "pre_train", "post_train"],
                        help="Which responses to evaluate: initial test, "
                             "pre-train (online), or post-train (online)")
    return parser.parse_args()


def init_client(api_provider):
    urls = {
        "together": "https://api.together.xyz/v1",
        "sambanova": "https://api.sambanova.ai/v1",
        "openai": "https://api.openai.com/v1",
    }
    keys = {
        "together": os.getenv("TOGETHER_API_KEY", ""),
        "sambanova": os.getenv("SAMBANOVA_API_KEY", ""),
        "openai": os.getenv("OPENAI_API_KEY", ""),
    }
    return openai.OpenAI(api_key=keys[api_provider], base_url=urls[api_provider])


def call_judge(client, model, question, reference_answers, should_abstain,
               model_answer, max_retries=3):
    if reference_answers:
        ref_str = "[" + "; ".join(str(a) for a in reference_answers) + "]"
    else:
        ref_str = "[]"

    prompt = LLM_JUDGE_ABSTENTION_PROMPT.format(
        question=question,
        ref_answer=ref_str,
        abstention_label=should_abstain,
        model_answer=model_answer,
    )

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0,
            )
            answer = resp.choices[0].message.content.lower().strip(" .,\n")
            if answer in ["yes", "no"]:
                return answer == "yes"
        except Exception as e:
            time.sleep(2 ** attempt)
    return None


def _parse_timestamp(log, fpath):
    ts = log.get("timestamp")
    if isinstance(ts, (int, float)):
        return float(ts)

    dt_str = log.get("datetime")
    if isinstance(dt_str, str):
        try:
            return datetime.fromisoformat(dt_str).timestamp()
        except ValueError:
            pass

    return os.path.getmtime(fpath)


def _extract_log_index(call_id, eval_type):
    if eval_type == "pre_train":
        match = re.search(r"online_train_s_(\d+)_gen_initial(?:_|$)", call_id)
        if match:
            return int(match.group(1)) - 1, "pre_train"
        return None

    if eval_type == "post_train":
        match = re.search(r"online_train_s_(\d+)_post_curate(?:_|$)", call_id)
        if match:
            return int(match.group(1)) - 1, "post_train"
        return None

    if eval_type == "initial":
        if match := re.search(r"^initial_test_eval_(\d+)$", call_id):
            return int(match.group(1)), "initial_test_eval"

        if match := re.search(r"^test_eval_(\d+)$", call_id):
            return int(match.group(1)), "legacy_test_eval"

        if match := re.search(r"test_eval_(\d+)$", call_id):
            source = "window_test_eval" if call_id.startswith("window_") else "other_test_eval"
            return int(match.group(1)), source

    return None


def _source_priority(eval_type, source):
    if eval_type != "initial":
        return 0

    priority = {
        "initial_test_eval": 0,
        "legacy_test_eval": 1,
        "other_test_eval": 2,
        "window_test_eval": 3,
    }
    return priority.get(source, 99)


def load_responses_from_logs(log_dir, pattern, eval_type, expected_total=None):
    """Load and de-duplicate model responses from detailed_llm_logs."""
    files = sorted(glob.glob(os.path.join(log_dir, pattern)))
    candidates = {}
    parsed_logs = 0

    for fpath in files:
        with open(fpath) as f:
            log = json.load(f)

        call_id = log.get("call_id", "")
        parsed = _extract_log_index(call_id, eval_type)
        if parsed is None:
            continue

        idx, source = parsed
        if idx < 0:
            continue

        parsed_logs += 1
        candidates.setdefault(idx, []).append(
            {
                "response": log.get("response", ""),
                "source": source,
                "timestamp": _parse_timestamp(log, fpath),
                "call_id": call_id,
                "path": fpath,
            }
        )

    responses = {}
    selected_entries = {}
    duplicate_count = 0

    for idx, entries in candidates.items():
        if len(entries) > 1:
            duplicate_count += len(entries) - 1
        entries.sort(
            key=lambda e: (
                _source_priority(eval_type, e["source"]),
                e["timestamp"],
                e["path"],
            )
        )
        selected = entries[0]
        responses[idx] = selected["response"]
        selected_entries[idx] = {
            "call_id": selected["call_id"],
            "source": selected["source"],
            "path": selected["path"],
        }

    missing_indices = []
    if expected_total is not None:
        missing_indices = [i for i in range(expected_total) if i not in responses]

    stats = {
        "glob_pattern": pattern,
        "files_scanned": len(files),
        "parsed_logs": parsed_logs,
        "selected_count": len(responses),
        "duplicate_count": duplicate_count,
        "missing_count": len(missing_indices),
        "missing_indices": missing_indices,
        "selected_entries": selected_entries,
    }
    return responses, stats


def main():
    args = parse_args()
    client = init_client(args.api_provider)

    # Load original data
    raw_data = load_data(args.data_path)
    processor = DataProcessor(task_name="eval")
    processed = processor.process_task_data(raw_data)

    log_dir = os.path.join(args.results_dir, "detailed_llm_logs")

    # Determine which logs to load based on eval_type
    if args.eval_type == "initial":
        pattern = "generator_*test_eval_*.json"
    elif args.eval_type == "pre_train":
        pattern = "generator_online_train_s_*_gen_initial_*.json"
    elif args.eval_type == "post_train":
        pattern = "generator_online_train_s_*_post_curate_*.json"
    else:
        raise ValueError(f"Unknown eval_type: {args.eval_type}")

    print(f"Loading responses from: {pattern}")
    responses, load_stats = load_responses_from_logs(
        log_dir=log_dir,
        pattern=pattern,
        eval_type=args.eval_type,
        expected_total=len(processed),
    )
    print(
        f"Found {load_stats['selected_count']} responses "
        f"(parsed={load_stats['parsed_logs']}, duplicates_collapsed={load_stats['duplicate_count']})"
    )
    print(
        f"Coverage: {len(processed) - load_stats['missing_count']}/{len(processed)} "
        f"(missing={load_stats['missing_count']})"
    )

    if not responses:
        print("No responses found. Check results_dir and eval_type.")
        return

    # Evaluate with LLM Judge
    results = []

    def evaluate_one(idx):
        if idx >= len(processed):
            return None
        sample = processed[idx]
        full_response = responses[idx]
        others = sample.get("others", {})

        is_abstention = call_judge(
            client, args.judge_model,
            question=sample["question"],
            reference_answers=others.get("reference_answers"),
            should_abstain=others.get("should_abstain", False),
            model_answer=full_response,
        )

        should_abstain = others.get("should_abstain", False)
        ground_truth = sample["target"]

        if is_abstention is None:
            correct = None
        elif ground_truth == "ABSTAIN":
            correct = is_abstention
        else:
            correct = not is_abstention
            # Could add answer correctness check here if needed

        return {
            "index": idx,
            "should_abstain": should_abstain,
            "judge_says_abstention": is_abstention,
            "correct": correct,
        }

    print(f"\nEvaluating {len(responses)} samples with LLM Judge ({args.max_workers} workers)...")
    indices = sorted(responses.keys())

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(evaluate_one, idx): idx for idx in indices}
        done = 0
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)
            done += 1
            if done % 50 == 0:
                print(f"  Progress: {done}/{len(indices)}")

    # Compute metrics
    results.sort(key=lambda x: x["index"])

    valid = [r for r in results if r["correct"] is not None]
    indeterminate = len(results) - len(valid)
    tp = sum(1 for r in valid if r["should_abstain"] and r["judge_says_abstention"])
    fn = sum(1 for r in valid if r["should_abstain"] and not r["judge_says_abstention"])
    fp = sum(1 for r in valid if not r["should_abstain"] and r["judge_says_abstention"])
    tn = sum(1 for r in valid if not r["should_abstain"] and not r["judge_says_abstention"])

    evaluated_total = len(valid)
    correct = sum(1 for r in valid if r["correct"])
    accuracy = correct / evaluated_total if evaluated_total else 0

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    print(f"\n{'='*60}")
    print(f"LLM JUDGE EVALUATION RESULTS ({args.eval_type})")
    print(f"{'='*60}")
    print(f"Total samples: {len(results)}")
    print(f"Evaluated: {evaluated_total}")
    print(f"Indeterminate: {indeterminate}")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{evaluated_total})")
    print(f"")
    print(f"Confusion Matrix:")
    print(f"  TP (should abstain, did abstain):     {tp}")
    print(f"  FN (should abstain, did NOT abstain): {fn}")
    print(f"  FP (should answer, but abstained):    {fp}")
    print(f"  TN (should answer, did answer):       {tn}")
    print(f"")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"{'='*60}")

    # Save results
    output = {
        "eval_type": args.eval_type,
        "judge_model": args.judge_model,
        "data_path": args.data_path,
        "total": len(results),
        "evaluated_total": evaluated_total,
        "indeterminate": indeterminate,
        "accuracy": accuracy,
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1,
        "log_selection": load_stats,
        "timestamp": datetime.now().isoformat(),
        "details": results,
    }
    out_path = os.path.join(
        args.results_dir,
        f"llm_judge_eval_{args.eval_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
