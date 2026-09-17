"""
TRM-UI sharded TAOBAO-MM vocabulary builder.

Purpose
-------
Build an exact frequency-based vocabulary from the full TAOBAO-MM train split
without the slow "resume by replaying all previous samples" problem.

Design
------
- Uses the raw Hugging Face streaming dataset directly.
- Uses the dataset's 161 physical shards as checkpoint boundaries.
- Processes shards independently.
- Never converts records into PyTorch tensors.
- Writes bounded sorted frequency chunks while processing a shard.
- Checkpoints completed shards atomically.
- A failed/interrupted run only needs to redo the currently incomplete shard.
- Final vocabulary is selected by exact global occurrence count.
- Ties are deterministic: lower raw ID wins.
- 0 = PAD, 1..N = known IDs, N+1 = UNK.

Important
---------
This builder intentionally does NOT use dataset.taobao_dataset.get_taobao_dataset().
That loader converts raw records into model tensors, which is unnecessary for
vocabulary construction.

The builder uses only:
    150_2_180  history item IDs
    151_2_180  history category IDs
    205        target item ID
    206        target category ID

Default:
    max items      = 18,000,000
    max categories = 100,000
    chunk size     = 10,000 samples per local frequency chunk
    workers        = 4

The worker implementation uses IterableDataset.shard(), so each worker gets a
disjoint subset of the physical dataset shards. Each shard has its own output
directory and completion marker.
"""

from __future__ import annotations

import argparse
import heapq
import json
import os
import struct
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

# ---------------------------------------------------------------------------
# Dataset fields
# ---------------------------------------------------------------------------

DATASET_NAME = "TaoBao-MM/Taobao-MM"
DEFAULT_SPLIT = "train"

HISTORY_ITEM_FIELD = "150_2_180"
HISTORY_CATEGORY_FIELD = "151_2_180"
TARGET_ITEM_FIELD = "205"
TARGET_CATEGORY_FIELD = "206"

# Binary record: signed 64-bit raw ID + unsigned 64-bit frequency.
RECORD = struct.Struct("<qQ")
RECORD_SIZE = RECORD.size

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MAX_ITEMS = 18_000_000
DEFAULT_MAX_CATEGORIES = 100_000
DEFAULT_CHUNK_SAMPLES = 10_000
DEFAULT_WORKERS = 4

# A bounded Counter prevents a worker from growing without limit.
DEFAULT_MAX_COUNTER_ENTRIES = 2_000_000

# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def atomic_json_write(path: Path, payload: dict) -> None:
    """Atomically replace a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")

    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp, path)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Raw field helpers
# ---------------------------------------------------------------------------


def unwrap_scalar(value):
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return value[0]
    return value


def iter_ids(value) -> Iterator[int]:
    """Yield integer IDs from a scalar/list/tensor-like value."""
    if value is None:
        return

    # We deliberately avoid torch imports in this builder. TAOBAO-MM raw
    # streaming records normally contain Python lists for these fields.
    if hasattr(value, "tolist"):
        value = value.tolist()

    if isinstance(value, (list, tuple)):
        for x in value:
            yield int(x)
    else:
        yield int(value)


def record_ids(record: dict) -> Tuple[Iterator[int], Iterator[int]]:
    """
    Return item/category ID iterators for one raw TAOBAO-MM record.

    Target IDs are included in the returned iterators.
    """
    item_values = iter_ids(record[HISTORY_ITEM_FIELD])
    category_values = iter_ids(record[HISTORY_CATEGORY_FIELD])

    target_item = int(unwrap_scalar(record[TARGET_ITEM_FIELD]))
    target_category = int(unwrap_scalar(record[TARGET_CATEGORY_FIELD]))

    def items() -> Iterator[int]:
        yield from item_values
        yield target_item

    def categories() -> Iterator[int]:
        yield from category_values
        yield target_category

    return items(), categories()


# ---------------------------------------------------------------------------
# Frequency chunk I/O
# ---------------------------------------------------------------------------


def write_frequency_chunk(
    counter: Counter,
    path: Path,
) -> None:
    """
    Write one sorted frequency chunk.

    Each record is:
        raw_id:int64
        count:uint64
    """
    if not counter:
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "wb") as f:
        for raw_id, count in sorted(counter.items()):
            f.write(RECORD.pack(int(raw_id), int(count)))
        f.flush()
        os.fsync(f.fileno())


def read_frequency_chunk(path: Path) -> Iterator[Tuple[int, int]]:
    with open(path, "rb") as f:
        while True:
            data = f.read(RECORD_SIZE)
            if not data:
                break
            if len(data) != RECORD_SIZE:
                raise RuntimeError(f"Corrupt frequency chunk: {path}")
            yield RECORD.unpack(data)


def merge_two_counters(
    counter_a: Counter,
    counter_b: Counter,
) -> Counter:
    result = Counter(counter_a)
    result.update(counter_b)
    return result


# ---------------------------------------------------------------------------
# Shard worker
# ---------------------------------------------------------------------------


def process_one_physical_shard(
    shard_index: int,
    num_shards: int,
    split: str,
    resume_root: str,
    chunk_samples: int,
    max_counter_entries: int,
) -> dict:
    """
    Process exactly one physical TAOBAO-MM shard.

    The shard is independently checkpointable. The worker writes:
        shard_xxxxxx/items_*.bin
        shard_xxxxxx/categories_*.bin
        shard_xxxxxx/complete.json
    """
    from datasets import load_dataset

    shard_dir = Path(resume_root) / f"shard_{shard_index:06d}"
    shard_dir.mkdir(parents=True, exist_ok=True)

    complete_path = shard_dir / "complete.json"

    if complete_path.exists():
        state = load_json(complete_path)
        return {
            "shard": shard_index,
            "status": "already_complete",
            "samples": int(state.get("samples", 0)),
            "item_chunks": state.get("item_chunks", []),
            "category_chunks": state.get("category_chunks", []),
        }

    # Streaming dataset. The .shard() operation partitions physical shards;
    # it does not require a sample -> byte offset mapping.
    ds = load_dataset(
        DATASET_NAME,
        split=split,
        streaming=True,
    )

    shard_ds = ds.shard(
        num_shards=num_shards,
        index=shard_index,
        contiguous=True,
    )

    item_counter: Counter = Counter()
    category_counter: Counter = Counter()

    item_chunks: List[str] = []
    category_chunks: List[str] = []

    samples = 0
    chunk_id = 0

    started = time.time()

    def flush_chunk() -> None:
        nonlocal chunk_id

        if not item_counter and not category_counter:
            return

        item_name = f"items_{chunk_id:08d}.bin"
        category_name = f"categories_{chunk_id:08d}.bin"

        item_path = shard_dir / item_name
        category_path = shard_dir / category_name

        write_frequency_chunk(item_counter, item_path)
        write_frequency_chunk(category_counter, category_path)

        item_chunks.append(item_name)
        category_chunks.append(category_name)

        # Clear only after both files have been safely written.
        item_counter.clear()
        category_counter.clear()
        chunk_id += 1

    for record in shard_ds:
        items, categories = record_ids(record)

        for raw_id in items:
            item_counter[raw_id] += 1

        for raw_id in categories:
            category_counter[raw_id] += 1

        samples += 1

        # Chunk boundaries are SAMPLE boundaries, not occurrence boundaries.
        if samples % chunk_samples == 0:
            flush_chunk()

        # Counter size is a safety valve. It does not change correctness:
        # flushing creates another exact frequency chunk.
        if (
            len(item_counter) >= max_counter_entries
            or len(category_counter) >= max_counter_entries
        ):
            flush_chunk()

        if samples % (chunk_samples * 10) == 0:
            elapsed = max(time.time() - started, 1e-9)
            rate = samples / elapsed
            print(
                f"[SHARD {shard_index:03d}] "
                f"samples={samples:,} rate={rate:,.0f}/s",
                flush=True,
            )

    flush_chunk()

    state = {
        "version": 1,
        "shard": shard_index,
        "num_shards": num_shards,
        "split": split,
        "status": "complete",
        "samples": samples,
        "item_chunks": item_chunks,
        "category_chunks": category_chunks,
    }

    atomic_json_write(complete_path, state)

    elapsed = max(time.time() - started, 1e-9)
    print(
        f"[SHARD {shard_index:03d}] COMPLETE "
        f"samples={samples:,} "
        f"rate={samples / elapsed:,.0f}/s",
        flush=True,
    )

    return {
        "shard": shard_index,
        "status": "complete",
        "samples": samples,
        "item_chunks": item_chunks,
        "category_chunks": category_chunks,
    }


# ---------------------------------------------------------------------------
# Global checkpoint
# ---------------------------------------------------------------------------


def build_initial_checkpoint(
    split: str,
    num_shards: int,
    max_items: int,
    max_categories: int,
    chunk_samples: int,
    workers: int,
) -> dict:
    return {
        "version": 1,
        "status": "scanning",
        "dataset": DATASET_NAME,
        "split": split,
        "num_shards": num_shards,
        "max_items": max_items,
        "max_categories": max_categories,
        "chunk_samples": chunk_samples,
        "workers": workers,
        "completed_shards": [],
        "item_chunks": [],
        "category_chunks": [],
    }


def scan_all_shards(
    split: str,
    resume_root: Path,
    checkpoint_path: Path,
    max_items: int,
    max_categories: int,
    chunk_samples: int,
    workers: int,
    max_counter_entries: int,
    shard_start: int = 0,
    shard_end: int | None = None,
) -> Tuple[List[str], List[str], int]:
    """
    Scan all physical shards.

    Completed shards are never submitted again. The global checkpoint is
    updated after every completed shard.
    """
    from datasets import load_dataset

    probe = load_dataset(
        DATASET_NAME,
        split=split,
        streaming=True,
    )

    num_shards = int(probe.num_shards)

    if num_shards <= 0:
        raise RuntimeError("TAOBAO-MM reported zero physical shards.")

    print("=" * 72)
    print("TRM-UI SHARDED TAOBAO-MM VOCABULARY BUILDER")
    print("=" * 72)
    print(f"Dataset:             {DATASET_NAME}")
    print(f"Split:               {split}")
    print(f"Physical shards:     {num_shards}")
    print(f"Max item IDs:        {max_items:,}")
    print(f"Max category IDs:    {max_categories:,}")
    if shard_end is None:
        shard_end = num_shards - 1

    if not (0 <= shard_start <= shard_end < num_shards):
        raise RuntimeError(
            f"Invalid shard range {shard_start}-{shard_end}; "
            f"valid range is 0-{num_shards - 1}."
        )

    selected_shards = list(range(shard_start, shard_end + 1))

    print(f"Chunk size:          {chunk_samples:,} samples")
    print(f"Workers:             {workers}")
    print(f"Shard range:         {shard_start:03d}-{shard_end:03d} "
          f"({len(selected_shards)} shards)")
    print(f"Resume directory:    {resume_root}")
    print()

    config = {
        "dataset": DATASET_NAME,
        "split": split,
        "num_shards": num_shards,
        "max_items": max_items,
        "max_categories": max_categories,
        "chunk_samples": chunk_samples,
        "workers": workers,
        "max_counter_entries": max_counter_entries,
    }

    if checkpoint_path.exists():
        checkpoint = load_json(checkpoint_path)

        old_config = {
            "dataset": checkpoint.get("dataset"),
            "split": checkpoint.get("split"),
            "num_shards": checkpoint.get("num_shards"),
            "max_items": checkpoint.get("max_items"),
            "max_categories": checkpoint.get("max_categories"),
            "chunk_samples": checkpoint.get("chunk_samples"),
            "workers": checkpoint.get("workers"),
            "max_counter_entries": checkpoint.get(
                "max_counter_entries",
                max_counter_entries,
            ),
        }

        # workers may safely change between runs.
        old_config.pop("workers", None)
        new_config = dict(config)
        new_config.pop("workers", None)

        if old_config != new_config:
            raise RuntimeError(
                "Existing checkpoint configuration does not match.\n"
                f"Existing: {old_config}\n"
                f"Current:  {new_config}\n"
                "Use --reset only if you intentionally want to start over."
            )
    else:
        checkpoint = build_initial_checkpoint(
            split=split,
            num_shards=num_shards,
            max_items=max_items,
            max_categories=max_categories,
            chunk_samples=chunk_samples,
            workers=workers,
        )
        checkpoint["max_counter_entries"] = max_counter_entries
        atomic_json_write(checkpoint_path, checkpoint)

    completed = set(int(x) for x in checkpoint.get("completed_shards", []))

    # Also trust per-shard completion markers. This makes the builder robust
    # if Colab died after writing a shard marker but before updating global
    # checkpoint.json.
    for shard_index in selected_shards:
        marker = resume_root / f"shard_{shard_index:06d}" / "complete.json"
        if marker.exists():
            completed.add(shard_index)

    selected_completed = sorted(
        shard_index for shard_index in selected_shards
        if shard_index in completed
    )

    if selected_completed:
        print(
            f"Already completed in selected range: "
            f"{len(selected_completed)}/{len(selected_shards)} shards"
        )

    remaining = [
        shard_index
        for shard_index in selected_shards
        if shard_index not in completed
    ]

    if not remaining:
        print("All physical shards are already complete.")
    else:
        print(f"Remaining shards:   {len(remaining)}")

        # Each process loads only its assigned physical shard subset.
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    process_one_physical_shard,
                    shard_index,
                    num_shards,
                    split,
                    str(resume_root),
                    chunk_samples,
                    max_counter_entries,
                ): shard_index
                for shard_index in remaining
            }

            for future in as_completed(futures):
                shard_index = futures[future]
                result = future.result()

                completed.add(shard_index)
                completed_list = sorted(completed)

                checkpoint["completed_shards"] = completed_list
                atomic_json_write(checkpoint_path, checkpoint)

                selected_done = sum(
                    1 for x in selected_shards if x in completed
                )
                print(
                    f"[CHECKPOINT] completed "
                    f"{selected_done}/{len(selected_shards)} selected shards "
                    f"(last={shard_index})",
                    flush=True,
                )

    # Collect chunk lists from completion markers. This avoids relying on
    # worker return order.
    item_chunks: List[str] = []
    category_chunks: List[str] = []
    total_samples = 0

    # In distributed --scan-only mode, collect only the selected range.
    # In normal full-scan mode, selected_shards is 0..num_shards-1.
    for shard_index in selected_shards:
        marker = resume_root / f"shard_{shard_index:06d}" / "complete.json"
        if not marker.exists():
            raise RuntimeError(
                f"Missing completion marker for selected shard {shard_index}."
            )

        state = load_json(marker)
        total_samples += int(state["samples"])

        shard_dir = resume_root / f"shard_{shard_index:06d}"

        for name in state.get("item_chunks", []):
            item_chunks.append(str(shard_dir / name))

        for name in state.get("category_chunks", []):
            category_chunks.append(str(shard_dir / name))

    checkpoint["status"] = "scan_complete"
    checkpoint["completed_shards"] = sorted(completed)
    checkpoint["item_chunks"] = item_chunks
    checkpoint["category_chunks"] = category_chunks
    checkpoint["total_samples"] = total_samples
    atomic_json_write(checkpoint_path, checkpoint)

    print()
    print(
        f"[SCAN COMPLETE] {total_samples:,} samples across "
        f"{num_shards} shards"
    )
    print(f"Item chunks:     {len(item_chunks):,}")
    print(f"Category chunks: {len(category_chunks):,}")
    print()

    return item_chunks, category_chunks, total_samples


# ---------------------------------------------------------------------------
# Exact global merge and top-K
# ---------------------------------------------------------------------------


def iter_merged_counts(paths: Sequence[Path]) -> Iterator[Tuple[int, int]]:
    """
    Exact k-way merge of sorted frequency chunks.

    Each input chunk contains sorted raw IDs. Equal IDs across chunks are
    summed before yielding.
    """
    streams = [iter(read_frequency_chunk(path)) for path in paths]

    heap: List[Tuple[int, int, int]] = []

    for stream_index, stream in enumerate(streams):
        try:
            raw_id, count = next(stream)
            heapq.heappush(heap, (raw_id, count, stream_index))
        except StopIteration:
            pass

    while heap:
        raw_id, count, stream_index = heapq.heappop(heap)
        total = count

        try:
            next_raw_id, next_count = next(streams[stream_index])
            heapq.heappush(
                heap,
                (next_raw_id, next_count, stream_index),
            )
        except StopIteration:
            pass

        while heap and heap[0][0] == raw_id:
            _, same_count, same_stream_index = heapq.heappop(heap)
            total += same_count

            try:
                next_raw_id, next_count = next(streams[same_stream_index])
                heapq.heappush(
                    heap,
                    (next_raw_id, next_count, same_stream_index),
                )
            except StopIteration:
                pass

        yield raw_id, total


def select_top_k(
    chunk_paths: Sequence[Path],
    max_known: int,
    progress_name: str,
) -> List[Tuple[int, int]]:
    """
    Select exact top-K by:
        1. descending total frequency
        2. ascending raw ID for ties

    We retain only K candidates in memory.
    """
    if not chunk_paths:
        raise RuntimeError(f"No {progress_name} frequency chunks found.")

    print(
        f"[SELECT] Exact global merge for {progress_name}: "
        f"{len(chunk_paths):,} chunks"
    )

    # Min-heap containing the current best K entries.
    # Heap key is (frequency, -raw_id, raw_id), so the worst entry is at root.
    best: List[Tuple[int, int, int]] = []

    merged = 0
    started = time.time()

    for raw_id, total_count in iter_merged_counts(
        [Path(p) for p in chunk_paths]
    ):
        merged += 1

        key = (total_count, -raw_id, raw_id)

        if len(best) < max_known:
            heapq.heappush(best, key)
        elif key > best[0]:
            heapq.heapreplace(best, key)

        if merged % 1_000_000 == 0:
            elapsed = max(time.time() - started, 1e-9)
            print(
                f"[SELECT] {progress_name}: "
                f"unique IDs={merged:,} "
                f"rate={merged / elapsed:,.0f}/s",
                flush=True,
            )

    ranked = [
        (raw_id, count)
        for count, neg_raw_id, raw_id in best
    ]

    ranked.sort(
        key=lambda pair: (-pair[1], pair[0])
    )

    print(
        f"[SELECT] {progress_name}: retained "
        f"{len(ranked):,} IDs"
    )

    return ranked


# ---------------------------------------------------------------------------
# Vocabulary output
# ---------------------------------------------------------------------------


def save_vocab(
    ranked: Sequence[Tuple[int, int]],
    path: Path,
    metadata: dict,
) -> Dict[str, int]:
    """
    Save raw-ID -> compact embedding index mapping.

    JSON object keys are strings because JSON object keys must be strings.
    """
    mapping = {
        str(raw_id): index
        for index, (raw_id, _) in enumerate(ranked, start=1)
    }

    payload = {
        "mapping": mapping,
        "metadata": metadata,
    }

    atomic_json_write(path, payload)

    return mapping


# ---------------------------------------------------------------------------
# Main build
# ---------------------------------------------------------------------------


def build_vocab(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    resume_root = Path(args.resume_dir)
    checkpoint_path = resume_root / "checkpoint.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    resume_root.mkdir(parents=True, exist_ok=True)

    if args.reset:
        print(f"Resetting resume directory: {resume_root}")
        for child in resume_root.iterdir():
            if child.is_dir():
                import shutil
                shutil.rmtree(child)
            else:
                child.unlink()

    if args.merge_only:
        # Merge-only mode is deliberately strict: after collecting shard
        # directories from multiple runtimes/accounts, we must never silently
        # rescan a missing shard.
        from datasets import load_dataset

        probe = load_dataset(
            DATASET_NAME,
            split=args.split,
            streaming=True,
        )
        num_shards = int(probe.num_shards)

        missing = []
        item_chunks = []
        category_chunks = []
        total_samples = 0

        for shard_index in range(num_shards):
            shard_dir = resume_root / f"shard_{shard_index:06d}"
            marker = shard_dir / "complete.json"
            if not marker.exists():
                missing.append(shard_index)
                continue

            state = load_json(marker)
            total_samples += int(state.get("samples", 0))
            item_chunks.extend(str(shard_dir / name)
                               for name in state.get("item_chunks", []))
            category_chunks.extend(str(shard_dir / name)
                                   for name in state.get("category_chunks", []))

        if missing:
            preview = ", ".join(f"{x:03d}" for x in missing[:20])
            suffix = " ..." if len(missing) > 20 else ""
            raise RuntimeError(
                f"Merge-only mode found {len(missing)} missing shard(s): "
                f"{preview}{suffix}. No vocabulary was generated."
            )

        print("=" * 72)
        print("MERGE-ONLY MODE")
        print("=" * 72)
        print(f"All {num_shards} shard completion markers found.")
        print(f"Samples represented: {total_samples:,}")
        print(f"Item chunks: {len(item_chunks):,}")
        print(f"Category chunks: {len(category_chunks):,}")
        print()

    else:
        item_chunks, category_chunks, total_samples = scan_all_shards(
            split=args.split,
            resume_root=resume_root,
            checkpoint_path=checkpoint_path,
            max_items=args.max_items,
            max_categories=args.max_categories,
            chunk_samples=args.chunk_samples,
            workers=args.workers,
            max_counter_entries=args.max_counter_entries,
            shard_start=args.shard_start,
            shard_end=args.shard_end,
        )

        if args.scan_only:
            print()
            print("SCAN-ONLY COMPLETE")
            print(f"Scanned shards: {args.shard_start:03d}-"
                  f"{args.shard_end if args.shard_end is not None else 'LAST'}")
            print("Final vocabulary merge was intentionally skipped.")
            return

        # A partial shard range must never accidentally produce a partial
        # vocabulary. Final vocabulary construction is only valid after all
        # physical shards are available.
        if args.shard_start != 0 or args.shard_end is not None:
            raise RuntimeError(
                "Final vocabulary construction requires the full shard range "
                "0..160. Use --scan-only for distributed shard scanning, "
                "then use --merge-only after gathering all shard directories."
            )

    # Final selection is deterministic and exact.
    ranked_items = select_top_k(
        item_chunks,
        args.max_items,
        "items",
    )

    ranked_categories = select_top_k(
        category_chunks,
        args.max_categories,
        "categories",
    )

    if not ranked_items:
        raise RuntimeError("No item IDs found.")

    if not ranked_categories:
        raise RuntimeError("No category IDs found.")

    item_metadata = {
        "builder": "TRM-UI sharded exact frequency top-K",
        "dataset": DATASET_NAME,
        "split": args.split,
        "num_shards": 161,
        "total_samples": total_samples,
        "max_known": args.max_items,
        "known_ids": len(ranked_items),
        "selection": "descending exact occurrence count; ties by ascending raw ID",
        "pad_index": 0,
        "unk_index": len(ranked_items) + 1,
        "vocab_rows": len(ranked_items) + 2,
    }

    category_metadata = {
        "builder": "TRM-UI sharded exact frequency top-K",
        "dataset": DATASET_NAME,
        "split": args.split,
        "num_shards": 161,
        "total_samples": total_samples,
        "max_known": args.max_categories,
        "known_ids": len(ranked_categories),
        "selection": "descending exact occurrence count; ties by ascending raw ID",
        "pad_index": 0,
        "unk_index": len(ranked_categories) + 1,
        "vocab_rows": len(ranked_categories) + 2,
    }

    save_vocab(
        ranked_items,
        output_dir / "item_vocab.json",
        item_metadata,
    )

    save_vocab(
        ranked_categories,
        output_dir / "category_vocab.json",
        category_metadata,
    )

    stats = {
        "builder": "TRM-UI sharded exact frequency top-K",
        "dataset": DATASET_NAME,
        "split": args.split,
        "num_shards": 161,
        "total_samples": total_samples,
        "max_items": args.max_items,
        "max_categories": args.max_categories,
        "known_items": len(ranked_items),
        "known_categories": len(ranked_categories),
        "item_vocab_rows": len(ranked_items) + 2,
        "category_vocab_rows": len(ranked_categories) + 2,
        "item_unk_index": len(ranked_items) + 1,
        "category_unk_index": len(ranked_categories) + 1,
        "item_chunks": len(item_chunks),
        "category_chunks": len(category_chunks),
    }

    atomic_json_write(
        output_dir / "vocab_stats.json",
        stats,
    )

    checkpoint = load_json(checkpoint_path)
    checkpoint["status"] = "complete"
    checkpoint["output_dir"] = str(output_dir)
    atomic_json_write(checkpoint_path, checkpoint)

    print()
    print("=" * 72)
    print("VOCABULARY COMPLETE")
    print("=" * 72)
    print(f"Samples scanned:      {total_samples:,}")
    print(f"Known items:          {len(ranked_items):,}")
    print(f"Item embedding rows:  {len(ranked_items) + 2:,}")
    print(f"Known categories:     {len(ranked_categories):,}")
    print(f"Category rows:        {len(ranked_categories) + 2:,}")
    print(f"Output:               {output_dir}")
    print(f"Resume data:          {resume_root}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build exact TAOBAO-MM vocabulary using physical shards."
    )

    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        choices=["train", "test"],
    )

    parser.add_argument(
        "--max-items",
        type=int,
        default=DEFAULT_MAX_ITEMS,
    )

    parser.add_argument(
        "--max-categories",
        type=int,
        default=DEFAULT_MAX_CATEGORIES,
    )

    parser.add_argument(
        "--chunk-samples",
        type=int,
        default=DEFAULT_CHUNK_SAMPLES,
        help="Flush frequency chunks every N dataset samples.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help="Number of physical-shard worker processes.",
    )

    parser.add_argument(
        "--shard-start",
        type=int,
        default=0,
        help="First physical shard to scan (inclusive).",
    )

    parser.add_argument(
        "--shard-end",
        type=int,
        default=None,
        help="Last physical shard to scan (inclusive). Defaults to the last shard.",
    )

    parser.add_argument(
        "--scan-only",
        action="store_true",
        help="Scan only the selected shard range and do not build the final vocabulary.",
    )

    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Do not scan. Require all 161 shard completion markers, then build the final vocabulary.",
    )

    parser.add_argument(
        "--max-counter-entries",
        type=int,
        default=DEFAULT_MAX_COUNTER_ENTRIES,
        help="Safety limit for unique IDs held by one worker.",
    )

    parser.add_argument(
        "--output-dir",
        default="vocab/dev",
    )

    parser.add_argument(
        "--resume-dir",
        default="vocab/resume_sharded",
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the sharded resume state and start from scratch.",
    )

    args = parser.parse_args()

    if args.scan_only and args.merge_only:
        parser.error("--scan-only and --merge-only cannot be used together")

    if args.shard_start < 0:
        parser.error("--shard-start must be >= 0")

    if args.shard_end is not None and args.shard_end < args.shard_start:
        parser.error("--shard-end must be >= --shard-start")

    if args.max_items <= 0:
        parser.error("--max-items must be positive")

    if args.max_categories <= 0:
        parser.error("--max-categories must be positive")

    if args.chunk_samples <= 0:
        parser.error("--chunk-samples must be positive")

    if args.workers <= 0:
        parser.error("--workers must be positive")

    if args.max_counter_entries <= 0:
        parser.error("--max-counter-entries must be positive")

    build_vocab(args)


if __name__ == "__main__":
    main()
