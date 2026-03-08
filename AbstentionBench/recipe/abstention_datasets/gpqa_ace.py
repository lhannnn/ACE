"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from recipe.abstention_datasets.abstract_abstention_dataset import (
    AbstentionDataset,
    Prompt,
)


class GPQAACE(AbstentionDataset):
    """Loads GPQA prompts exported by ACE from a JSONL file."""

    def __init__(
        self,
        data_path="/Users/luohan/Desktop/ACE/ace/eval/abstention/data/gpqa_all.jsonl",
        max_num_samples=None,
    ):
        super().__init__()
        self.data_path = data_path
        self.max_num_samples = max_num_samples
        self.dataset = self._load_data()

    def _load_data(self) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        with open(Path(self.data_path), "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    def __len__(self):
        if self.max_num_samples is None:
            return len(self.dataset)
        return min(self.max_num_samples, len(self.dataset))

    def __getitem__(self, idx) -> Prompt:
        if idx >= len(self):
            raise IndexError

        item = self.dataset[idx]
        question: str = item["question"]
        reference_answers: Optional[List[str]] = item.get("reference_answers")
        should_abstain: bool = item["should_abstain"]
        metadata: Dict[str, Any] = item.get("metadata", {})

        return Prompt(
            question=question,
            reference_answers=reference_answers,
            should_abstain=should_abstain,
            metadata=metadata,
        )
