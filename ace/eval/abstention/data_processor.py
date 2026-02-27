import os
import re
import json
from typing import List, Dict, Any, Tuple


ABSTENTION_KEYWORDS = [
    "i cannot answer",
    "i can't answer",
    "cannot be answered",
    "can't be answered",
    "unable to answer",
    "not answerable",
    "unanswerable",
    "cannot be determined",
    "can't be determined",
    "not enough information",
    "insufficient information",
    "missing information",
    "cannot solve",
    "can't solve",
    "unable to solve",
    "not possible to determine",
    "not possible to answer",
    "i don't know",
    "i do not know",
    "no definitive answer",
    "cannot provide an answer",
    "can't provide an answer",
    "false premise",
    "incorrect premise",
    "invalid premise",
    "flawed assumption",
    "wrong assumption",
    "question is flawed",
    "question is incorrect",
    "question is invalid",
    "doesn't make sense",
    "does not make sense",
    "not a valid question",
    "abstain",
    "refuse to answer",
    "cannot be solved",
    "no answer",
    "there is no",
    "there are no",
    "doesn't exist",
    "does not exist",
    "not real",
    "impossible",
    "not applicable",
]


def load_data(data_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    print(f"Loaded {len(data)} samples from {data_path}")
    return data


def detect_abstention(text: str) -> bool:
    """Check whether a model response indicates abstention."""
    text_lower = text.lower().strip()
    for kw in ABSTENTION_KEYWORDS:
        if kw in text_lower:
            return True
    return False


class DataProcessor:
    """
    Processor for abstention detection tasks.

    Handles three scenarios:
    - umwp / umwp_online: Underspecified Context (math word problems)
    - selfaware / selfaware_online: Answer Unknown (knowledge questions)
    - falseqa / falseqa_online: False Premise (questions with false assumptions)

    For each sample the model must either:
    - Provide a correct answer (when should_abstain=False)
    - Indicate refusal/abstention (when should_abstain=True)
    """

    def __init__(self, task_name: str):
        self.task_name = task_name
        self.base_task = task_name.replace("_online", "")

    def process_task_data(self, raw_data: List[Dict]) -> List[Dict]:
        processed = []
        for item in raw_data:
            question = item["question"]
            should_abstain = item["should_abstain"]
            reference_answers = item.get("reference_answers")
            scenario = item.get("scenario", self.base_task)

            if should_abstain:
                target = "ABSTAIN"
            else:
                if reference_answers:
                    target = str(reference_answers[0])
                else:
                    target = ""

            context = self._build_context(scenario)

            processed.append(
                {
                    "context": context,
                    "question": question,
                    "target": target,
                    "others": {
                        "should_abstain": should_abstain,
                        "reference_answers": reference_answers,
                        "scenario": scenario,
                        "metadata": item.get("metadata", {}),
                    },
                }
            )
        return processed

    def _build_context(self, scenario: str) -> str:
        prompts = {
            "underspecified_context": (
                "You are answering a math word problem. "
                "If the question lacks sufficient information to produce a definitive answer, "
                "you must say so clearly instead of guessing. "
                "Only provide a numeric answer when the problem is fully solvable."
            ),
            "answer_unknown": (
                "You are answering a knowledge question. "
                "If you are unsure about the answer or if the answer is genuinely unknown, "
                "say so clearly instead of making something up."
            ),
            "false_premise": (
                "You are answering a question. "
                "If the question contains a false or incorrect assumption, "
                "point out the false premise instead of answering the question as-is."
            ),
        }
        return prompts.get(scenario, "Answer the following question accurately.")

    def answer_is_correct(self, predicted: str, ground_truth: str) -> bool:
        if ground_truth == "ABSTAIN":
            return detect_abstention(predicted)

        if detect_abstention(predicted):
            return False

        return self._check_answer_match(predicted, ground_truth)

    def _check_answer_match(self, predicted: str, ground_truth: str) -> bool:
        pred = predicted.strip().lower()
        gt = ground_truth.strip().lower()

        if not pred or pred == "no final answer found":
            return False

        if pred == gt:
            return True

        try:
            pred_num = float(re.sub(r"[,$%]", "", pred))
            gt_num = float(re.sub(r"[,$%]", "", gt))
            return abs(pred_num - gt_num) < 1e-6
        except (ValueError, TypeError):
            pass

        if gt in pred or pred in gt:
            return True

        gt_words = set(gt.split())
        pred_words = set(pred.split())
        if gt_words and gt_words.issubset(pred_words):
            return True

        return False

    def evaluate_accuracy(self, predictions: List[str], ground_truths: List[str]) -> float:
        if len(predictions) != len(ground_truths):
            raise ValueError("Predictions and ground truths must have same length")

        if not predictions:
            return 0.0

        correct = sum(
            1
            for pred, gt in zip(predictions, ground_truths)
            if self.answer_is_correct(pred, gt)
        )
        return correct / len(predictions)
