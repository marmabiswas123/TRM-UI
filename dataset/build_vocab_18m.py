"""
Resumable, exact top-K frequency vocabulary builder for TAOBAO-MM.

Key properties
--------------
1. Scans the selected dataset split(s) completely.
2. Counts item/category occurrences exactly.
3. Keeps at most K known items (default 18M) and K categories.
4. Writes progress continuously to disk.
5. If interrupted, resumes from the last completed dataset sample instead of
   starting the scan again.
6. Item/category frequency chunks are durable on disk; completed chunks are
   never rescanned.
7. Vocabulary selection is performed only after the complete scan finishes.

For a clean train/test benchmark, use:
    --splits train

The resume state is stored inside:
    <output-dir>/_resume/

Delete that directory (or use --reset) only if the underlying dataset or
scan configuration has changed.
"""

from __future__ import annotations

import argparse
import heapq
import json
import shutil
import struct
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np

from dataset.taobao_dataset import get_taobao_dataset
from dataset.taobao_features import IDMapper


# raw ID = signed int64, count = unsigned uint64
_RECORD = struct.Struct("<qQ")
_RECORD_SIZE = _RECORD.size


def _write_sorted_counts(values: list[int], path: Path) -> None:
    """Sort one chunk and write exact per-ID counts."""
    if not values:
        return

    arr = np.asarray(values, dtype=np.int64)
    arr.sort()

    change = np.empty(arr.size, dtype=np.bool_)
    change[0] = True
    change[1:] = arr[1:] != arr[:-1]

    starts = np.flatnonzero(change)
    ends = np.r_[starts[1:], arr.size]

    with path.open("wb") as f:
        for start, end in zip(starts, ends):
            f.write(_RECORD.pack(int(arr[start]), int(end - start)))


def _iter_records(path: Path) -> Iterable[tuple[int, int]]:
    with path.open("rb") as f:
        while True:
            data = f.read(_RECORD_SIZE)
            if not data:
                return
            if len(data) != _RECORD_SIZE:
                raise IOError(f"Corrupt frequency chunk: {path}")
            yield _RECORD.unpack(data)


def _merge_frequency_chunks(
    chunk_paths: list[Path],
    top_k: int,
) -> list[tuple[int, int]]:
    """Exact k-way merge and top-K selection."""
    if not chunk_paths:
        return []

    streams = [iter(_iter_records(p)) for p in chunk_paths]
    merge_heap: list[tuple[int, int, int]] = []

    for idx, stream in enumerate(streams):
        try:
            raw_id, count = next(stream)
            heapq.heappush(merge_heap, (raw_id, count, idx))
        except StopIteration:
            pass

    # (count, -raw_id, raw_id)
    # Equal counts prefer smaller raw IDs.
    selected: list[tuple[int, int, int]] = []

    while merge_heap:
        raw_id, count, stream_idx = heapq.heappop(merge_heap)

        total = count
        same_streams = [stream_idx]

        while merge_heap and merge_heap[0][0] == raw_id:
            _, more_count, other_idx = heapq.heappop(merge_heap)
            total += more_count
            same_streams.append(other_idx)

        for idx in same_streams:
            try:
                next_id, next_count = next(streams[idx])
                heapq.heappush(merge_heap, (next_id, next_count, idx))
            except StopIteration:
                pass

        candidate = (total, -raw_id, raw_id)
        if len(selected) < top_k:
            heapq.heappush(selected, candidate)
        elif candidate > selected[0]:
            heapq.heapreplace(selected, candidate)

    selected.sort(key=lambda x: (-x[0], x[2]))
    return [(raw_id, count) for count, _, raw_id in selected]


def _atomic_json_write(path: Path, payload: dict) -> None:
    """Write JSON atomically so an interrupted write cannot corrupt state."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2)
        f.flush()
    tmp.replace(path)


def _sample_ids(sample) -> tuple[list[int], list[int]]:
    items = [int(x) for x in sample["history_items"].tolist()]
    items.append(int(sample["target_item"].item()))

    categories = [int(x) for x in sample["history_categories"].tolist()]
    categories.append(int(sample["target_category"].item()))

    return items, categories


def _scan_split_resumable(
    split: str,
    history_length: int,
    chunk_size: int,
    resume_dir: Path,
    start_sample: int,
    item_chunks: list[str],
    category_chunks: list[str],
    next_chunk_id: int,
) -> tuple[int, list[str], list[str], int]:
    """
    Scan one split.

    Progress is committed only after:
      1. the completed chunk(s) have been fsynced/closed, and
      2. the checkpoint has been atomically replaced.

    Chunks end on sample boundaries, so resume never starts in the middle of
    a sample.
    """
    dataset = get_taobao_dataset(
        split=split,
        history_length=history_length,
    )

    items_buffer: list[int] = []
    categories_buffer: list[int] = []
    sample_index = 0
    processed = start_sample

    # Skip exactly the samples already committed in the checkpoint.
    for sample in dataset:
        if sample_index < start_sample:
            sample_index += 1
            continue

        items, categories = _sample_ids(sample)
        items_buffer.extend(items)
        categories_buffer.extend(categories)
        sample_index += 1
        processed += 1

        # Flush at sample boundaries. This makes the checkpoint exact.
        if len(items_buffer) >= chunk_size:
            item_path = resume_dir / f"items_{next_chunk_id:08d}.bin"
            category_path = resume_dir / f"categories_{next_chunk_id:08d}.bin"

            _write_sorted_counts(items_buffer, item_path)
            _write_sorted_counts(categories_buffer, category_path)

            # Ensure chunk files are physically flushed before checkpointing.
            with item_path.open("ab") as f:
                f.flush()
            with category_path.open("ab") as f:
                f.flush()

            item_chunks.append(item_path.name)
            category_chunks.append(category_path.name)
            next_chunk_id += 1

            items_buffer.clear()
            categories_buffer.clear()

            checkpoint = {
                "version": 2,
                "status": "scanning",
                "config": {
                    "version": 2,
                    "splits": [split],
                    "max_items": 18_000_000,
                    "max_categories": 100_000,
                    "history_length": history_length,
                },
                "split": split,
                "processed_samples": processed,
                "item_chunks": item_chunks,
                "category_chunks": category_chunks,
                "next_chunk_id": next_chunk_id,
            }
            _atomic_json_write(resume_dir / "checkpoint.json", checkpoint)

            if len(item_chunks) % 10 == 0:
                print(
                    f"  {split}: checkpoint at {processed:,} samples | "
                    f"{len(item_chunks):,} frequency chunks"
                )

    # Flush the final partial chunk, also at a sample boundary.
    if items_buffer:
        item_path = resume_dir / f"items_{next_chunk_id:08d}.bin"
        category_path = resume_dir / f"categories_{next_chunk_id:08d}.bin"

        _write_sorted_counts(items_buffer, item_path)
        _write_sorted_counts(categories_buffer, category_path)

        with item_path.open("ab") as f:
            f.flush()
        with category_path.open("ab") as f:
            f.flush()

        item_chunks.append(item_path.name)
        category_chunks.append(category_path.name)
        next_chunk_id += 1

    checkpoint = {
        "version": 2,
        "status": "split_complete",
        "split": split,
        "processed_samples": processed,
        "item_chunks": item_chunks,
        "category_chunks": category_chunks,
        "next_chunk_id": next_chunk_id,
    }
    _atomic_json_write(resume_dir / "checkpoint.json", checkpoint)

    return processed, item_chunks, category_chunks, next_chunk_id


def _mapper_from_ranked(
    ranked: list[tuple[int, int]],
    max_size: int,
) -> IDMapper:
    mapper = IDMapper(max_size=max_size)
    for raw_id, _count in ranked:
        mapper.add(raw_id)
    return mapper


def _save_mapper(mapper: IDMapper, path: Path, metadata: dict) -> None:
    payload = mapper.state_dict()
    payload["_build_metadata"] = metadata
    _atomic_json_write(path, payload)


def build_vocab(
    splits: list[str],
    max_items: int,
    max_categories: int,
    history_length: int,
    chunk_size: int,
    output_dir: str,
    reset: bool = False,
) -> None:
    if max_items <= 0:
        raise ValueError("--max-items must be positive")
    if max_categories <= 0:
        raise ValueError("--max-categories must be positive")
    if chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    resume_dir = output_path / "_resume"

    if reset and resume_dir.exists():
        shutil.rmtree(resume_dir)

    resume_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = resume_dir / "checkpoint.json"

    # These parameters define the meaning of the vocabulary scan and must
    # remain unchanged when resuming. chunk_size is deliberately excluded:
    # it only controls when accumulated occurrences are spilled to disk.
    config = {
        "version": 2,
        "splits": splits,
        "max_items": max_items,
        "max_categories": max_categories,
        "history_length": history_length,
    }

    checkpoint = None
    if checkpoint_path.exists():
        with checkpoint_path.open() as f:
            checkpoint = json.load(f)

        old_config = checkpoint.get("config")
        if old_config is not None and old_config != config:
            raise RuntimeError(
                "Existing resume checkpoint was created with different "
                "configuration.\n"
                f"Existing: {old_config}\n"
                f"Current:  {config}\n"
                "Use --reset only if you intentionally want to restart."
            )

        # Older checkpoints without config are not safe to reuse.
        if old_config is None:
            raise RuntimeError(
                "Existing checkpoint has no configuration metadata. "
                "Use --reset to start a clean resumable scan."
            )

    if checkpoint is None:
        checkpoint = {
            "version": 2,
            "status": "starting",
            "config": config,
            "split_index": 0,
            "processed_samples": 0,
            "item_chunks": [],
            "category_chunks": [],
            "next_chunk_id": 0,
        }
        _atomic_json_write(checkpoint_path, checkpoint)

    # The global chunk lists are kept across split boundaries.
    item_chunks = list(checkpoint.get("item_chunks", []))
    category_chunks = list(checkpoint.get("category_chunks", []))
    next_chunk_id = int(checkpoint.get("next_chunk_id", 0))
    split_index = int(checkpoint.get("split_index", 0))
    processed_samples = int(checkpoint.get("processed_samples", 0))

    print("TRM-UI RESUMABLE FREQUENCY VOCABULARY BUILDER")
    print("=" * 64)
    print(f"Splits:                  {', '.join(splits)}")
    print(f"Maximum item vocabulary: {max_items:,}")
    print(f"Maximum category vocab:  {max_categories:,}")
    print(f"History length:          {history_length}")
    print(f"Chunk size:              {chunk_size:,} occurrences")
    print(f"Output:                  {output_path}")
    print()
    print("Each completed chunk is written to disk and checkpointed immediately.")
    print("Chunk size is approximately 200,000 occurrences by default.")
    print("Progress is checkpointed continuously.")
    print("Restarting this command resumes from the last committed sample.")
    print()

    # -----------------------------
    # Exact scan with resume
    # -----------------------------
    for idx in range(split_index, len(splits)):
        split = splits[idx]

        # If resuming a split, processed_samples tells us where to continue.
        # For a new split, start at sample 0.
        start_sample = processed_samples if idx == split_index else 0

        print(
            f"[SCAN] {split}: "
            f"resuming after {start_sample:,} samples"
            if start_sample
            else f"[SCAN] {split}: starting from sample 0"
        )

        processed, item_chunks, category_chunks, next_chunk_id = (
            _scan_split_resumable(
                split=split,
                history_length=history_length,
                chunk_size=chunk_size,
                resume_dir=resume_dir,
                start_sample=start_sample,
                item_chunks=item_chunks,
                category_chunks=category_chunks,
                next_chunk_id=next_chunk_id,
            )
        )

        split_index = idx + 1
        processed_samples = 0

        checkpoint = {
            "version": 2,
            "status": "scanning",
            "config": config,
            "split_index": split_index,
            "processed_samples": processed_samples,
            "item_chunks": item_chunks,
            "category_chunks": category_chunks,
            "next_chunk_id": next_chunk_id,
        }
        _atomic_json_write(checkpoint_path, checkpoint)

        print(f"[SCAN] {split}: complete ({processed:,} samples)")
        print()

    # -----------------------------
    # Exact top-K selection
    # -----------------------------
    print("[SELECT] Merging item frequency chunks...")
    item_paths = [resume_dir / name for name in item_chunks]
    ranked_items = _merge_frequency_chunks(item_paths, max_items)

    print("[SELECT] Merging category frequency chunks...")
    category_paths = [resume_dir / name for name in category_chunks]
    ranked_categories = _merge_frequency_chunks(
        category_paths,
        max_categories,
    )

    if not ranked_items:
        raise RuntimeError("No item IDs were found.")
    if not ranked_categories:
        raise RuntimeError("No category IDs were found.")

    item_mapper = _mapper_from_ranked(ranked_items, max_items)
    category_mapper = _mapper_from_ranked(
        ranked_categories,
        max_categories,
    )

    item_cutoff = ranked_items[-1][1] if len(ranked_items) == max_items else 0
    category_cutoff = (
        ranked_categories[-1][1]
        if len(ranked_categories) == max_categories
        else 0
    )

    metadata = {
        "builder": "TRM-UI resumable exact frequency top-K",
        "splits": splits,
        "history_length": history_length,
        "max_items": max_items,
        "max_categories": max_categories,
        "chunk_size": chunk_size,
        "item_known": len(item_mapper),
        "category_known": len(category_mapper),
        "item_cutoff_frequency": item_cutoff,
        "category_cutoff_frequency": category_cutoff,
        "item_selection": "top-K by exact occurrence count; ties by raw ID",
        "category_selection": "top-K by exact occurrence count; ties by raw ID",
        "resume": True,
    }

    _save_mapper(
        item_mapper,
        output_path / "item_vocab.json",
        metadata,
    )
    _save_mapper(
        category_mapper,
        output_path / "category_vocab.json",
        metadata,
    )

    stats = {
        **metadata,
        "item_vocab_rows": item_mapper.vocab_size,
        "category_vocab_rows": category_mapper.vocab_size,
        "item_unk_index": item_mapper.unk_index,
        "category_unk_index": category_mapper.unk_index,
    }
    _atomic_json_write(output_path / "vocab_stats.json", stats)

    checkpoint = {
        "version": 2,
        "status": "complete",
        "config": config,
        "split_index": len(splits),
        "processed_samples": 0,
        "item_chunks": item_chunks,
        "category_chunks": category_chunks,
        "next_chunk_id": next_chunk_id,
    }
    _atomic_json_write(checkpoint_path, checkpoint)

    print()
    print("=" * 64)
    print("VOCABULARY COMPLETE")
    print("=" * 64)
    print(f"Known items:           {len(item_mapper):,}")
    print(f"Item embedding rows:   {item_mapper.vocab_size:,}")
    print(f"Item cutoff frequency:  {item_cutoff:,}")
    print(f"Known categories:      {len(category_mapper):,}")
    print(f"Category rows:         {category_mapper.vocab_size:,}")
    print(f"Category cutoff:       {category_cutoff:,}")
    print(f"Saved to:              {output_path}")
    print()
    print("Resume data retained at:")
    print(f"  {resume_dir}")
    print("You may keep it for reproducibility or delete it after verification.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a resumable exact top-K TAOBAO-MM vocabulary."
    )

    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train"],
        choices=["train", "test"],
        help="Split(s) used to build the vocabulary. Default: train.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=18_000_000,
        help="Maximum known item IDs. Default: 18M.",
    )
    parser.add_argument(
        "--max-categories",
        type=int,
        default=100_000,
        help="Maximum known category IDs. Default: 100k.",
    )
    parser.add_argument(
        "--history-length",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200_000,
        help="Approximate occurrence count per durable checkpoint chunk. Each completed chunk is immediately written and checkpointed.",
    )
    parser.add_argument(
        "--output-dir",
        default="vocab/train_18m",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Discard existing resume state and start from sample 0.",
    )

    args = parser.parse_args()

    build_vocab(
        splits=args.splits,
        max_items=args.max_items,
        max_categories=args.max_categories,
        history_length=args.history_length,
        chunk_size=args.chunk_size,
        output_dir=args.output_dir,
        reset=args.reset,
    )


if __name__ == "__main__":
    main()
