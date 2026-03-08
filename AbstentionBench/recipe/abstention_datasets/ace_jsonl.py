"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import json
from pathlib import Path
from typing import Any

from recipe.abstention_datasets.abstract_abstention_dataset import (
    AbstentionDataset,
    Prompt,
)


class ACEJSONLDataset(AbstentionDataset):
    """Generic loader for ACE-style abstention JSONL files."""

    def __init__(
        self,
        data_path: str,
        max_num_samples: int | None = None,
    ):
        super().__init__()
        if not data_path:
            raise ValueError("data_path is required for ACEJSONLDataset")

        self.data_path = Path(data_path)
        self.max_num_samples = max_num_samples
        self.dataset = self._load_data()

    def _load_data(self) -> list[dict[str, Any]]:
        if not self.data_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {self.data_path}")

        records: list[dict[str, Any]] = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records

    def __len__(self) -> int:
        if self.max_num_samples is None:
            return len(self.dataset)
        return min(self.max_num_samples, len(self.dataset))

    def __getitem__(self, idx) -> Prompt:
        if idx >= len(self):
            raise IndexError

        item = self.dataset[idx]
        return Prompt(
            question=item["question"],
            reference_answers=item.get("reference_answers"),
            should_abstain=item.get("should_abstain"),
            metadata=item.get("metadata", {}),
        )
