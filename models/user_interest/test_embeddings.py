"""
Integration test for the TRM-UI embedding layer + TAOBAO-MM loader.
"""

import torch

from dataset.taobao_dataset import get_taobao_dataset
from models.user_interest.embeddings import UserInterestEmbedding
from dataset.taobao_features import build_sample_vocabularies
from dataset.taobao_features import remap_sample


def main():
    print("TAOBAO-MM → TRM-UI embedding integration test")
    print("=" * 60)

    # ------------------------------------------------------------
    # Load one real TAOBAO-MM sample
    # ------------------------------------------------------------

    dataset = get_taobao_dataset(
        split="train",
        history_length=50,
    )

    sample = next(iter(dataset))

    print("\nRaw TRM-UI sample:")
    for key, value in sample.items():
        print(
            f"  {key:22s}"
            f" shape={tuple(value.shape)!s:10s}"
            f" dtype={value.dtype}"
        )

    # ------------------------------------------------------------
    # Add batch dimension
    # ------------------------------------------------------------

    batch = {
        key: value.unsqueeze(0)
        for key, value in sample.items()
    }

    # ------------------------------------------------------------
    # Remap raw TAOBAO IDs to consecutive embedding indices
    # ------------------------------------------------------------

    item_mapper, category_mapper = build_sample_vocabularies(
        sample
    )

    sample = remap_sample(
        sample,
        item_mapper,
        category_mapper,
    )

    batch = {
        key: value.unsqueeze(0)
        for key, value in sample.items()
    }

    num_items = item_mapper.vocab_size
    num_categories = category_mapper.vocab_size

    print("\nVocabulary sizes:")
    print("  items:      ", num_items)
    print("  categories: ", num_categories)

    # ------------------------------------------------------------
    # Create embedding model
    # ------------------------------------------------------------

    model = UserInterestEmbedding(
        num_items=num_items,
        num_categories=num_categories,
        item_embedding_dim=128,
        category_embedding_dim=128,
    )

    # ------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------

    with torch.no_grad():
        output = model(
            history_items=batch["history_items"],
            history_categories=batch["history_categories"],
            target_item=batch["target_item"],
            target_category=batch["target_category"],
        )

    # ------------------------------------------------------------
    # Validate output
    # ------------------------------------------------------------

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

    print("\n✓ Real TAOBAO-MM sample successfully passed through embeddings.")
    print("✓ Shapes are correct.")
    print("✓ No NaN/Inf detected.")


if __name__ == "__main__":
    main()
