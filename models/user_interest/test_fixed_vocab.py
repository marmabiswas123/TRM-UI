"""
Integration test for:

TAOBAO-MM
    -> fixed vocabulary
    -> TRM-UI embedding layer
"""

import torch

from dataset.taobao_dataset import get_taobao_dataset
from models.user_interest.embeddings import UserInterestEmbedding


ITEM_VOCAB = "vocab/dev/item_vocab.json"
CATEGORY_VOCAB = "vocab/dev/category_vocab.json"


def main():
    print("TAOBAO-MM -> fixed vocabulary -> TRM-UI")
    print("=" * 60)

    dataset = get_taobao_dataset(
        split="train",
        history_length=50,
        item_vocab_path=ITEM_VOCAB,
        category_vocab_path=CATEGORY_VOCAB,
    )

    sample = next(iter(dataset))

    print("\nFixed-vocabulary sample:")

    for key, value in sample.items():
        preview = value[:10] if value.ndim > 0 else value.item()
        print(
            f"  {key:22s}"
            f" shape={tuple(value.shape)!s:10s}"
            f" dtype={value.dtype}"
            f" value={preview}"
        )

    # Add batch dimension.
    batch = {
        key: value.unsqueeze(0)
        for key, value in sample.items()
    }

    # The development vocabulary contains:
    #   items      = 38,319 known + PAD + UNK
    #   categories = 4,310 known + PAD + UNK
    num_items = 38_321
    num_categories = 4_312

    model = UserInterestEmbedding(
        num_items=num_items,
        num_categories=num_categories,
        item_embedding_dim=128,
        category_embedding_dim=128,
    )

    with torch.no_grad():
        output = model(
            history_items=batch["history_items"],
            history_categories=batch["history_categories"],
            target_item=batch["target_item"],
            target_category=batch["target_category"],
        )

    print("\nEmbedding output:")

    for key, value in output.items():
        print(
            f"  {key:22s}"
            f" shape={tuple(value.shape)!s:10s}"
            f" dtype={value.dtype}"
        )

    assert output["history"].shape == (1, 50, 256)
    assert output["target"].shape == (1, 256)

    assert torch.isfinite(output["history"]).all()
    assert torch.isfinite(output["target"]).all()

    assert int(batch["history_items"].max()) < num_items
    assert int(batch["history_categories"].max()) < num_categories
    assert int(batch["target_item"].max()) < num_items
    assert int(batch["target_category"].max()) < num_categories

    print("\n✓ Fixed vocabulary loaded.")
    print("✓ Raw IDs converted to stable embedding indices.")
    print("✓ Unknown IDs are safely representable by UNK.")
    print("✓ Embedding lookup succeeded.")
    print("✓ Shapes and numerical values are valid.")


if __name__ == "__main__":
    main()
