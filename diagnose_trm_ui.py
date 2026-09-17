"""
TRM-UI diagnostic experiment.

Purpose:
    1. Inspect label balance in the controlled TAOBAO-MM train/validation
       subsets.
    2. Measure fixed-vocabulary UNK/PAD coverage.
    3. Report sequence/target statistics.
    4. Optionally run a tiny overfit test to determine whether TRM-UI can
       memorize a small fixed training set.

This is a diagnostic tool, not the production trainer.

Expected project layout:

MyDrive/
├── TinyRecursiveModels/
└── TRM-UI/
    ├── trm_ui.py
    ├── diagnose_trm_ui.py
    ├── models/
    │   └── user_interest/
    │       └── embedding.py
    ├── dataset/
    │   ├── taobao_dataset.py
    │   └── taobao_features.py
    └── vocab/
        └── dev/
            ├── item_vocab.json
            └── category_vocab.json
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn

from dataset.taobao_dataset import get_taobao_dataset
from dataset.taobao_features import load_mapper
from models.user_interest.embeddings import UserInterestEmbedding
from trm_ui import TRMUI


PROJECT_DIR = Path(__file__).resolve().parent
VOCAB_DIR = PROJECT_DIR / "vocab" / "dev"


# ---------------------------------------------------------------------------
# Dataset statistics
# ---------------------------------------------------------------------------

def inspect_split(
    split: str,
    max_samples: int,
    history_length: int,
    item_mapper,
    category_mapper,
):
    dataset = get_taobao_dataset(
        split=split,
        history_length=history_length,
        item_vocab_path=str(PROJECT_DIR / "vocab" / "dev" / "item_vocab.json"),
        category_vocab_path=str(
            PROJECT_DIR / "vocab" / "dev" / "category_vocab.json"
        ),
    )

    positives = 0
    negatives = 0

    item_unk = 0
    item_pad = 0
    category_unk = 0
    category_pad = 0

    history_positions = 0
    history_nonpad = 0

    target_item_unk = 0
    target_category_unk = 0

    unique_items = set()
    unique_categories = set()

    samples = 0
    start = time.time()

    for sample in dataset:
        labels = int(sample["label"].item())

        if labels == 1:
            positives += 1
        else:
            negatives += 1

        hi = sample["history_items"]
        hc = sample["history_categories"]
        ti = int(sample["target_item"].item())
        tc = int(sample["target_category"].item())

        item_unk += int((hi == item_mapper.unk_index).sum().item())
        item_pad += int((hi == 0).sum().item())
        category_unk += int(
            (hc == category_mapper.unk_index).sum().item()
        )
        category_pad += int((hc == 0).sum().item())

        history_positions += hi.numel()
        history_nonpad += int((hi != 0).sum().item())

        if ti == item_mapper.unk_index:
            target_item_unk += 1

        if tc == category_mapper.unk_index:
            target_category_unk += 1

        unique_items.update(
            int(x)
            for x in hi.tolist()
            if x != 0 and x != item_mapper.unk_index
        )
        unique_categories.update(
            int(x)
            for x in hc.tolist()
            if x != 0 and x != category_mapper.unk_index
        )

        samples += 1

        if samples >= max_samples:
            break

    elapsed = max(time.time() - start, 1e-9)

    total = positives + negatives

    return {
        "split": split,
        "samples": total,
        "positives": positives,
        "negatives": negatives,
        "positive_ratio": positives / total if total else float("nan"),
        "history_positions": history_positions,
        "history_nonpad": history_nonpad,
        "history_pad_rate": (
            1.0 - history_nonpad / history_positions
            if history_positions
            else float("nan")
        ),
        "history_item_unk_rate": (
            item_unk / history_positions
            if history_positions
            else float("nan")
        ),
        "history_category_unk_rate": (
            category_unk / history_positions
            if history_positions
            else float("nan")
        ),
        "target_item_unk_rate": (
            target_item_unk / total if total else float("nan")
        ),
        "target_category_unk_rate": (
            target_category_unk / total if total else float("nan")
        ),
        "unique_known_history_items": len(unique_items),
        "unique_known_history_categories": len(unique_categories),
        "samples_per_second": total / elapsed if total else 0.0,
    }


def print_stats(stats: Dict) -> None:
    print(f"\n{stats['split'].upper()} DATA")
    print("-" * 60)
    print(f"Samples:                       {stats['samples']:,}")
    print(f"Positive:                      {stats['positives']:,}")
    print(f"Negative:                      {stats['negatives']:,}")
    print(f"Positive ratio:                {stats['positive_ratio']:.4%}")
    print(f"Unique known history items:    {stats['unique_known_history_items']:,}")
    print(
        f"Unique known history categories:{stats['unique_known_history_categories']:>8,}"
    )
    print(f"History PAD rate:              {stats['history_pad_rate']:.4%}")
    print(f"History item UNK rate:         {stats['history_item_unk_rate']:.4%}")
    print(
        f"History category UNK rate:     "
        f"{stats['history_category_unk_rate']:.4%}"
    )
    print(f"Target item UNK rate:          {stats['target_item_unk_rate']:.4%}")
    print(
        f"Target category UNK rate:      "
        f"{stats['target_category_unk_rate']:.4%}"
    )
    print(f"Read throughput:               {stats['samples_per_second']:.2f} samples/s")


# ---------------------------------------------------------------------------
# Tiny overfit test
# ---------------------------------------------------------------------------

def collate(samples):
    return {
        "history_items": torch.stack([s["history_items"] for s in samples]),
        "history_categories": torch.stack(
            [s["history_categories"] for s in samples]
        ),
        "target_item": torch.stack([s["target_item"] for s in samples]),
        "target_category": torch.stack(
            [s["target_category"] for s in samples]
        ),
        "label": torch.stack([s["label"] for s in samples]).long(),
    }


def collect_fixed_samples(dataset, count: int):
    samples = []

    for sample in dataset:
        samples.append(
            {
                key: value.clone()
                for key, value in sample.items()
            }
        )

        if len(samples) >= count:
            break

    if not samples:
        raise RuntimeError("Dataset returned zero samples.")

    return samples


def run_overfit_test(args, device, item_mapper, category_mapper):
    print("\n" + "=" * 70)
    print("TINY OVERFIT TEST")
    print("=" * 70)

    dataset = get_taobao_dataset(
        split="train",
        history_length=args.history_length,
        item_vocab_path=str(args.item_vocab),
        category_vocab_path=str(args.category_vocab),
    )

    samples = collect_fixed_samples(dataset, args.overfit_samples)

    labels = torch.tensor(
        [int(s["label"].item()) for s in samples],
        dtype=torch.long,
    )

    positives = int(labels.sum().item())
    negatives = len(labels) - positives

    print(f"Fixed samples: {len(samples)}")
    print(f"Positive:      {positives}")
    print(f"Negative:      {negatives}")

    if positives == 0 or negatives == 0:
        raise RuntimeError(
            "Tiny overfit set contains only one class. "
            "Increase --overfit-samples."
        )

    embedding_model = UserInterestEmbedding(
        num_items=item_mapper.vocab_size,
        num_categories=category_mapper.vocab_size,
        item_embedding_dim=args.item_embed_dim,
        category_embedding_dim=args.category_embed_dim,
    ).to(device)

    trm_model = TRMUI(
        input_dim=args.item_embed_dim + args.category_embed_dim,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        expansion=4.0,
        history_length=args.history_length,
        H_cycles=args.h_cycles,
        L_cycles=args.l_cycles,
        L_layers=args.l_layers,
        forward_dtype="float32",
    ).to(device)

    optimizer = torch.optim.AdamW(
        list(embedding_model.parameters())
        + list(trm_model.parameters()),
        lr=args.overfit_lr,
        weight_decay=0.0,
    )

    embedding_model.train()
    trm_model.train()

    best_loss = float("inf")
    best_accuracy = 0.0

    for epoch in range(1, args.overfit_epochs + 1):
        random.shuffle(samples)

        total_loss = 0.0
        correct = 0
        total = 0

        for start in range(0, len(samples), args.overfit_batch_size):
            batch_samples = samples[
                start : start + args.overfit_batch_size
            ]

            batch = collate(batch_samples)

            history_items = batch["history_items"].to(device)
            history_categories = batch["history_categories"].to(device)
            target_item = batch["target_item"].to(device)
            target_category = batch["target_category"].to(device)
            labels_batch = batch["label"].float().to(device)

            embeddings = embedding_model(
                history_items=history_items,
                history_categories=history_categories,
                target_item=target_item,
                target_category=target_category,
            )

            outputs = trm_model(
                history_embeddings=embeddings["history"],
                target_embedding=embeddings["target"],
            )

            loss = nn.functional.binary_cross_entropy_with_logits(
                outputs["logits"],
                labels_batch,
            )

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss during overfit test.")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                list(embedding_model.parameters())
                + list(trm_model.parameters()),
                1.0,
            )

            optimizer.step()

            with torch.no_grad():
                predictions = (
                    outputs["probability"] >= 0.5
                ).long()

                correct += int(
                    (predictions == labels_batch.long()).sum().item()
                )

            batch_n = labels_batch.shape[0]
            total_loss += loss.item() * batch_n
            total += batch_n

        mean_loss = total_loss / total
        accuracy = correct / total

        best_loss = min(best_loss, mean_loss)
        best_accuracy = max(best_accuracy, accuracy)

        if epoch == 1 or epoch % args.overfit_log_every == 0:
            print(
                f"  epoch {epoch:4d} | "
                f"loss {mean_loss:.6f} | "
                f"accuracy {accuracy:.2%}"
            )

        # If the model memorizes the tiny set, we have established that the
        # optimization path and task interface have enough capacity.
        if mean_loss < args.overfit_stop_loss and accuracy >= 0.99:
            print(
                f"\n✓ Tiny dataset was memorized at epoch {epoch}."
            )
            break

    print("\nOverfit result:")
    print(f"  best loss:     {best_loss:.6f}")
    print(f"  best accuracy: {best_accuracy:.2%}")

    if best_accuracy >= 0.99 and best_loss < args.overfit_stop_loss:
        print(
            "✓ Optimization path can memorize a small fixed dataset."
        )
    else:
        print(
            "⚠ Tiny-set memorization was not reached. "
            "This warrants inspecting the model/task interface."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Diagnose TRM-UI data coverage and learning behavior."
    )

    parser.add_argument(
        "--mode",
        choices=["stats", "overfit", "all"],
        default="all",
    )

    parser.add_argument(
        "--item-vocab",
        type=Path,
        default=VOCAB_DIR / "item_vocab.json",
    )
    parser.add_argument(
        "--category-vocab",
        type=Path,
        default=VOCAB_DIR / "category_vocab.json",
    )

    parser.add_argument("--history-length", type=int, default=50)
    parser.add_argument("--train-samples", type=int, default=10_000)
    parser.add_argument("--val-samples", type=int, default=2_000)

    parser.add_argument("--overfit-samples", type=int, default=256)
    parser.add_argument("--overfit-epochs", type=int, default=100)
    parser.add_argument("--overfit-batch-size", type=int, default=16)
    parser.add_argument("--overfit-lr", type=float, default=3e-4)
    parser.add_argument("--overfit-stop-loss", type=float, default=0.02)
    parser.add_argument("--overfit-log-every", type=int, default=5)

    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--h-cycles", type=int, default=3)
    parser.add_argument("--l-cycles", type=int, default=6)
    parser.add_argument("--l-layers", type=int, default=2)
    parser.add_argument("--item-embed-dim", type=int, default=128)
    parser.add_argument("--category-embed-dim", type=int, default=128)

    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device)

    print("=" * 70)
    print("TRM-UI DATA + LEARNING DIAGNOSTICS")
    print("=" * 70)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU:    {torch.cuda.get_device_name(device)}")

    if not args.item_vocab.exists():
        raise FileNotFoundError(f"Missing item vocabulary: {args.item_vocab}")

    if not args.category_vocab.exists():
        raise FileNotFoundError(
            f"Missing category vocabulary: {args.category_vocab}"
        )

    item_mapper = load_mapper(args.item_vocab)
    category_mapper = load_mapper(args.category_vocab)

    print("\nVocabulary:")
    print(f"  item rows:       {item_mapper.vocab_size}")
    print(f"  item UNK index:  {item_mapper.unk_index}")
    print(f"  category rows:   {category_mapper.vocab_size}")
    print(f"  category UNK:    {category_mapper.unk_index}")

    if args.mode in ("stats", "all"):
        train_stats = inspect_split(
            "train",
            args.train_samples,
            args.history_length,
            item_mapper,
            category_mapper,
        )

        val_stats = inspect_split(
            "test",
            args.val_samples,
            args.history_length,
            item_mapper,
            category_mapper,
        )

        print_stats(train_stats)
        print_stats(val_stats)

        print("\nInterpretation hints:")
        print(
            "  - Very high UNK rates mean the limited dev vocabulary is "
            "hiding item/category identity."
        )
        print(
            "  - A strong train/validation label-ratio mismatch can distort "
            "the first controlled AUC result."
        )
        print(
            "  - Low UNK rates + balanced labels shift attention toward the "
            "model/task interface."
        )

    if args.mode in ("overfit", "all"):
        run_overfit_test(
            args,
            device,
            item_mapper,
            category_mapper,
        )

    print("\n" + "=" * 70)
    print("DIAGNOSTICS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
