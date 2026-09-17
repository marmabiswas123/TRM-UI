"""
TAOBAO-MM dataset loader for TRM-UI.

Converts the raw TAOBAO-MM / MUSE feature schema into the minimal
user-interest recommendation sample required by TRM-UI.

Raw fields used:
    label_0
    150_2_180
    151_2_180
    205
    206

Initially not used:
    129_1
    205_c
    150_2_180_c
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, Optional

import torch
from torch.utils.data import IterableDataset

from dataset.taobao_features import load_mapper, remap_sample


HISTORY_ITEM_FIELD = "150_2_180"
HISTORY_CATEGORY_FIELD = "151_2_180"

TARGET_ITEM_FIELD = "205"
TARGET_CATEGORY_FIELD = "206"

LABEL_FIELD = "label_0"

DEFAULT_HISTORY_LENGTH = 1000


def _unwrap_scalar(value: Any) -> Any:
    """Extract a scalar from a one-element list/tuple."""

    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]

    return value


def _extract_binary_label(value: Any) -> int:
    """
    Convert TAOBAO-MM's two-class label to an integer.

        [1, 0] -> 0
        [0, 1] -> 1
    """

    if isinstance(value, torch.Tensor):
        value = value.tolist()

    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(
                f"Expected a 2-class label, got: {value}"
            )

        if value[0] == 1 and value[1] == 0:
            return 0

        if value[0] == 0 and value[1] == 1:
            return 1

        raise ValueError(f"Invalid one-hot label: {value}")

    label = int(value)

    if label not in (0, 1):
        raise ValueError(f"Expected binary label 0/1, got {label}")

    return label


def _normalize_sequence(
    value: Any,
    max_length: int,
) -> list[int]:
    """
    Convert a history field into a fixed-length list.

    Longer sequences keep the most recent interactions.
    Shorter sequences are left-padded with 0.
    """

    if isinstance(value, torch.Tensor):
        value = value.tolist()

    if value is None:
        value = []

    value = [int(x) for x in list(value)]

    if len(value) > max_length:
        value = value[-max_length:]

    if len(value) < max_length:
        value = [0] * (max_length - len(value)) + value

    return value


def convert_taobao_record(
    record: Dict[str, Any],
    history_length: int = DEFAULT_HISTORY_LENGTH,
) -> Dict[str, torch.Tensor]:
    """Convert one raw TAOBAO-MM record into a TRM-UI sample."""

    required_fields = (
        HISTORY_ITEM_FIELD,
        HISTORY_CATEGORY_FIELD,
        TARGET_ITEM_FIELD,
        TARGET_CATEGORY_FIELD,
        LABEL_FIELD,
    )

    missing = [field for field in required_fields if field not in record]

    if missing:
        raise KeyError(
            f"TAOBAO-MM record is missing fields: {missing}"
        )

    history_items = _normalize_sequence(
        record[HISTORY_ITEM_FIELD],
        history_length,
    )

    history_categories = _normalize_sequence(
        record[HISTORY_CATEGORY_FIELD],
        history_length,
    )

    target_item = int(
        _unwrap_scalar(record[TARGET_ITEM_FIELD])
    )

    target_category = int(
        _unwrap_scalar(record[TARGET_CATEGORY_FIELD])
    )

    label = _extract_binary_label(record[LABEL_FIELD])

    return {
        "history_items": torch.tensor(
            history_items,
            dtype=torch.long,
        ),
        "history_categories": torch.tensor(
            history_categories,
            dtype=torch.long,
        ),
        "target_item": torch.tensor(
            target_item,
            dtype=torch.long,
        ),
        "target_category": torch.tensor(
            target_category,
            dtype=torch.long,
        ),
        "label": torch.tensor(
            label,
            dtype=torch.long,
        ),
    }


class TaobaoMMIterableDataset(IterableDataset):
    """Streaming TAOBAO-MM dataset with optional fixed vocabularies."""

    def __init__(
        self,
        split: str = "train",
        history_length: int = DEFAULT_HISTORY_LENGTH,
        shuffle: bool = False,
        seed: int = 0,
        item_vocab_path: Optional[str] = None,
        category_vocab_path: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.split = split
        self.history_length = history_length
        self.shuffle = shuffle
        self.seed = seed

        if (item_vocab_path is None) != (category_vocab_path is None):
            raise ValueError(
                "item_vocab_path and category_vocab_path must be "
                "provided together."
            )

        self.item_mapper = (
            load_mapper(item_vocab_path)
            if item_vocab_path is not None
            else None
        )

        self.category_mapper = (
            load_mapper(category_vocab_path)
            if category_vocab_path is not None
            else None
        )

        self._dataset = None

    def _load_dataset(self):
        """Lazily load the streaming Hugging Face dataset."""

        from datasets import load_dataset

        dataset = load_dataset(
            "TaoBao-MM/Taobao-MM",
            split=self.split,
            streaming=True,
        )

        if self.shuffle:
            dataset = dataset.shuffle(seed=self.seed)

        return dataset

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        if self._dataset is None:
            self._dataset = self._load_dataset()

        for record in self._dataset:
            sample = convert_taobao_record(
                record,
                history_length=self.history_length,
            )

            if (
                self.item_mapper is not None
                and self.category_mapper is not None
            ):
                sample = remap_sample(
                    sample,
                    self.item_mapper,
                    self.category_mapper,
                )

            yield sample


def get_taobao_dataset(
    split: str = "train",
    history_length: int = DEFAULT_HISTORY_LENGTH,
    shuffle: bool = False,
    seed: int = 0,
    item_vocab_path: Optional[str] = None,
    category_vocab_path: Optional[str] = None,
) -> TaobaoMMIterableDataset:
    """Construct a streaming TAOBAO-MM dataset."""

    return TaobaoMMIterableDataset(
        split=split,
        history_length=history_length,
        shuffle=shuffle,
        seed=seed,
        item_vocab_path=item_vocab_path,
        category_vocab_path=category_vocab_path,
    )


if __name__ == "__main__":
    dataset = get_taobao_dataset(
        split="train",
        history_length=50,
    )

    sample = next(iter(dataset))

    print("\nTRM-UI sample")
    print("=" * 60)

    for key, value in sample.items():
        preview = value[:10] if value.ndim > 0 else value.item()
        print(
            f"{key:22s}"
            f" shape={tuple(value.shape)!s:12s}"
            f" dtype={str(value.dtype):12s}"
            f" value={preview}"
        )
