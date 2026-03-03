#!/usr/bin/env python3
"""
Run ACE on abstention detection tasks (UMWP, SelfAware, FalseQA).

Usage examples:

  # Offline training on UMWP (train/val/test splits)
  python -m eval.abstention.run --task_name umwp --mode offline --save_path results

  # Online training on full UMWP dataset
  python -m eval.abstention.run --task_name umwp_online --mode online --save_path results

  # Eval-only with a saved playbook
  python -m eval.abstention.run --task_name umwp_online --mode eval_only \
      --initial_playbook_path results/ace_run_.../best_playbook.txt --save_path results
"""
import os
import json
import argparse
from datetime import datetime
from .data_processor import DataProcessor, load_data

from ace import ACE
from utils import initialize_clients


def parse_args():
    parser = argparse.ArgumentParser(description="Run ACE on abstention tasks")

    parser.add_argument(
        "--task_name",
        type=str,
        required=True,
        choices=["umwp", "selfaware", "falseqa", "coconot", "gpqa",
                 "umwp_online", "selfaware_online", "falseqa_online", "coconot_online", "gpqa_online"],
        help="Task name",
    )
    parser.add_argument("--initial_playbook_path", type=str, default=None)
    parser.add_argument(
        "--mode",
        type=str,
        default="online",
        choices=["offline", "online", "eval_only"],
    )

    parser.add_argument("--api_provider", type=str, default="together",
                        choices=["sambanova", "together", "openai"])
    parser.add_argument("--generator_model", type=str,
                        default="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
    parser.add_argument("--reflector_model", type=str,
                        default="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")
    parser.add_argument("--curator_model", type=str,
                        default="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo")

    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--max_num_rounds", type=int, default=3)
    parser.add_argument("--curator_frequency", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--online_eval_frequency", type=int, default=15)
    parser.add_argument("--save_steps", type=int, default=50)

    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--playbook_token_budget", type=int, default=80000)
    parser.add_argument("--test_workers", type=int, default=10)

    parser.add_argument("--json_mode", action="store_true")
    parser.add_argument("--no_ground_truth", action="store_true")
    parser.add_argument("--use_bulletpoint_analyzer", action="store_true")
    parser.add_argument("--bulletpoint_analyzer_threshold", type=float, default=0.90)

    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument(
        "--config_path",
        type=str,
        default="./eval/abstention/data/abstention_config.json",
    )

    return parser.parse_args()


def preprocess_data(task_name, config, mode, judge_client=None, judge_model=None):
    processor = DataProcessor(task_name=task_name,
                              judge_client=judge_client,
                              judge_model=judge_model)

    if mode in ["online", "eval_only"]:
        train_samples = None
        val_samples = None

        if "test_data" not in config:
            raise ValueError(f"{mode} mode requires test_data in config.")
        test_samples = load_data(config["test_data"])
        test_samples = processor.process_task_data(test_samples)

        label = "Online" if mode == "online" else "Eval only"
        print(f"{label} mode: {len(test_samples)} test examples")
    else:
        train_samples = load_data(config["train_data"])
        val_samples = load_data(config["val_data"])
        train_samples = processor.process_task_data(train_samples)
        val_samples = processor.process_task_data(val_samples)

        if "test_data" in config:
            test_samples = load_data(config["test_data"])
            test_samples = processor.process_task_data(test_samples)
        else:
            test_samples = []

        print(
            f"Offline mode: {len(train_samples)} train, "
            f"{len(val_samples)} val, {len(test_samples)} test"
        )

    return train_samples, val_samples, test_samples, processor


def load_initial_playbook(path):
    if path and os.path.exists(path):
        with open(path, "r") as f:
            return f.read()
    return None


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"ACE ABSTENTION EVALUATION")
    print(f"{'='*60}")
    print(f"Task: {args.task_name}")
    print(f"Mode: {args.mode.upper()}")
    print(f"Generator: {args.generator_model}")
    print(f"API Provider: {args.api_provider}")
    print(f"{'='*60}\n")

    with open(args.config_path, "r") as f:
        task_config = json.load(f)

    # Initialize a dedicated judge client for LLM-based abstention detection
    import openai as _openai
    provider_urls = {
        "together": "https://api.together.xyz/v1",
        "sambanova": "https://api.sambanova.ai/v1",
        "openai": "https://api.openai.com/v1",
    }
    provider_keys = {
        "together": os.getenv("TOGETHER_API_KEY", ""),
        "sambanova": os.getenv("SAMBANOVA_API_KEY", ""),
        "openai": os.getenv("OPENAI_API_KEY", ""),
    }
    judge_client = _openai.OpenAI(
        api_key=provider_keys[args.api_provider],
        base_url=provider_urls[args.api_provider],
    )
    print(f"LLM Judge initialized: {args.generator_model} via {args.api_provider}")

    train_samples, val_samples, test_samples, data_processor = preprocess_data(
        args.task_name, task_config[args.task_name], args.mode,
        judge_client=judge_client, judge_model=args.generator_model,
    )

    initial_playbook = load_initial_playbook(args.initial_playbook_path)
    if initial_playbook:
        print(f"Loaded initial playbook from {args.initial_playbook_path}\n")
    else:
        print("Using empty playbook as initial playbook\n")

    ace_system = ACE(
        api_provider=args.api_provider,
        generator_model=args.generator_model,
        reflector_model=args.reflector_model,
        curator_model=args.curator_model,
        max_tokens=args.max_tokens,
        initial_playbook=initial_playbook,
        use_bulletpoint_analyzer=args.use_bulletpoint_analyzer,
        bulletpoint_analyzer_threshold=args.bulletpoint_analyzer_threshold,
    )

    config = {
        "num_epochs": args.num_epochs,
        "max_num_rounds": args.max_num_rounds,
        "curator_frequency": args.curator_frequency,
        "eval_steps": args.eval_steps,
        "online_eval_frequency": args.online_eval_frequency,
        "save_steps": args.save_steps,
        "playbook_token_budget": args.playbook_token_budget,
        "task_name": args.task_name,
        "mode": args.mode,
        "json_mode": args.json_mode,
        "no_ground_truth": args.no_ground_truth,
        "save_dir": args.save_path,
        "test_workers": args.test_workers,
        "initial_playbook_path": args.initial_playbook_path,
        "use_bulletpoint_analyzer": args.use_bulletpoint_analyzer,
        "bulletpoint_analyzer_threshold": args.bulletpoint_analyzer_threshold,
        "api_provider": args.api_provider,
    }

    results = ace_system.run(
        mode=args.mode,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        data_processor=data_processor,
        config=config,
    )


if __name__ == "__main__":
    main()
