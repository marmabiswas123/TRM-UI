"""
TRM-UI training script.

First controlled experiment:
    TAOBAO-MM
        -> fixed vocabulary
        -> UserInterestEmbedding
        -> TRM-UI
        -> binary click prediction
        -> AdamW
        -> validation AUC
        -> Google Drive checkpoint

The trainer intentionally uses a bounded number of samples for the first
experiment. Once this is stable, increase the sample budget and history
length in controlled experiments.

Expected project layout:

MyDrive/
├── TinyRecursiveModels/
└── TRM-UI/
    ├── train_trm_ui.py
    ├── trm_ui.py
    ├── embeddings.py
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
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from torch import nn

from dataset.taobao_dataset import get_taobao_dataset
from dataset.taobao_features import load_mapper
from models.user_interest.embedding import UserInterestEmbedding
from trm_ui import TRMUI


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_HISTORY_LENGTH = 50
DEFAULT_TRAIN_SAMPLES = 10_000
DEFAULT_VAL_SAMPLES = 2_000
DEFAULT_BATCH_SIZE = 4
DEFAULT_EPOCHS = 3
DEFAULT_LR = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-2
DEFAULT_GRAD_CLIP = 1.0
DEFAULT_H_CYCLES = 3
DEFAULT_L_CYCLES = 6
DEFAULT_L_LAYERS = 2
DEFAULT_HIDDEN_SIZE = 512
DEFAULT_NUM_HEADS = 8
DEFAULT_ITEM_EMBED_DIM = 128
DEFAULT_CATEGORY_EMBED_DIM = 128
DEFAULT_SEED = 42

PROJECT_DIR = Path(__file__).resolve().parent
VOCAB_DIR = PROJECT_DIR / "vocab" / "dev"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # We prefer reproducibility for the first experiment.
    # These settings can be relaxed later for maximum throughput.
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Batch preparation
# ---------------------------------------------------------------------------

def collate_samples(samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Stack individual TAOBAO-MM samples into a batch."""
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


def make_batches(
    dataset: Iterable[Dict[str, torch.Tensor]],
    batch_size: int,
    max_samples: int,
):
    """
    Consume at most max_samples examples and yield mini-batches.

    The dataset is iterable because TAOBAO-MM is too large to materialize
    into memory.
    """
    batch: List[Dict[str, torch.Tensor]] = []
    count = 0

    for sample in dataset:
        batch.append(sample)
        count += 1

        if len(batch) == batch_size:
            yield collate_samples(batch)
            batch = []

        if count >= max_samples:
            break

    if batch:
        yield collate_samples(batch)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def binary_auc(labels: List[int], scores: List[float]) -> float:
    """
    Compute ROC-AUC without requiring sklearn.

    Returns NaN when the evaluation subset contains only one class.
    """
    if not labels:
        return float("nan")

    positives = sum(1 for y in labels if y == 1)
    negatives = len(labels) - positives

    if positives == 0 or negatives == 0:
        return float("nan")

    pairs = sorted(zip(scores, labels), key=lambda x: x[0])

    # Average ranks for tied scores.
    ranks = [0.0] * len(pairs)
    i = 0

    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1

        avg_rank = (i + 1 + j) / 2.0

        for k in range(i, j):
            ranks[k] = avg_rank

        i = j

    positive_rank_sum = sum(
        ranks[i] for i, (_, label) in enumerate(pairs) if label == 1
    )

    return (
        positive_rank_sum
        - positives * (positives + 1) / 2
    ) / (positives * negatives)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_models(args, device: torch.device):
    item_mapper = load_mapper(args.item_vocab)
    category_mapper = load_mapper(args.category_vocab)

    embedding_model = UserInterestEmbedding(
        num_items=item_mapper.vocab_size,
        num_categories=category_mapper.vocab_size,
        item_embedding_dim=args.item_embed_dim,
        category_embedding_dim=args.category_embed_dim,
        sparse_item_embedding=True,
        device=device,
    )

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

    return embedding_model, trm_model, item_mapper, category_mapper


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------

def forward_batch(
    batch: Dict[str, torch.Tensor],
    embedding_model: nn.Module,
    trm_model: nn.Module,
    device: torch.device,
):
    history_items = batch["history_items"].to(device, non_blocking=True)
    history_categories = batch["history_categories"].to(
        device, non_blocking=True
    )
    target_item = batch["target_item"].to(device, non_blocking=True)
    target_category = batch["target_category"].to(
        device, non_blocking=True
    )
    labels = batch["label"].float().to(device, non_blocking=True)

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
        labels,
    )

    return outputs, loss, labels


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_epoch(train_dataset,embedding_model,trm_model,item_optimizer,dense_optimizer,device,args,epoch:int):
    embedding_model.train(); trm_model.train()
    running_loss=0.0; examples=0; start_time=time.time()
    item_optimizer.zero_grad(set_to_none=True); dense_optimizer.zero_grad(set_to_none=True)
    dense_params=list(embedding_model.category_embedding.parameters())+list(trm_model.parameters())
    for step,batch in enumerate(make_batches(train_dataset,batch_size=args.batch_size,max_samples=args.train_samples),start=1):
        outputs,loss,labels=forward_batch(batch,embedding_model,trm_model,device)
        if not torch.isfinite(loss): raise RuntimeError(f'Non-finite training loss at epoch={epoch}, step={step}: {loss.item()}')
        loss.backward()
        grad_norm=torch.nn.utils.clip_grad_norm_(dense_params,args.grad_clip)
        if not torch.isfinite(torch.as_tensor(grad_norm)): raise RuntimeError(f'Non-finite dense gradient norm at epoch={epoch}, step={step}')
        item_optimizer.step(); dense_optimizer.step()
        item_optimizer.zero_grad(set_to_none=True); dense_optimizer.zero_grad(set_to_none=True)
        n=labels.shape[0]; running_loss+=loss.item()*n; examples+=n
        if step%args.log_every==0:
            elapsed=max(time.time()-start_time,1e-9); print(f'  epoch {epoch} | step {step:5d} | samples {examples:6d} | loss {running_loss/examples:.6f} | dense_grad_norm {float(grad_norm):.4f} | {examples/elapsed:.2f} samples/s')
    elapsed=max(time.time()-start_time,1e-9)
    return {'loss':running_loss/max(examples,1),'samples':examples,'seconds':elapsed,'samples_per_second':examples/elapsed}

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    val_dataset,
    embedding_model,
    trm_model,
    device,
    args,
):
    embedding_model.eval()
    trm_model.eval()

    losses = []
    labels_all: List[int] = []
    scores_all: List[float] = []

    examples = 0
    start_time = time.time()

    for batch in make_batches(
        val_dataset,
        batch_size=args.batch_size,
        max_samples=args.val_samples,
    ):
        outputs, loss, labels = forward_batch(
            batch,
            embedding_model,
            trm_model,
            device,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite validation loss: {loss.item()}"
            )

        probabilities = outputs["probability"]

        losses.append(loss.item() * labels.shape[0])
        labels_all.extend(labels.long().cpu().tolist())
        scores_all.extend(probabilities.float().cpu().tolist())
        examples += labels.shape[0]

    val_loss = sum(losses) / max(examples, 1)
    auc = binary_auc(labels_all, scores_all)

    elapsed = max(time.time() - start_time, 1e-9)

    return {
        "loss": val_loss,
        "auc": auc,
        "samples": examples,
        "seconds": elapsed,
    }


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(path:Path,epoch:int,embedding_model:nn.Module,trm_model:nn.Module,item_optimizer:torch.optim.Optimizer,dense_optimizer:torch.optim.Optimizer,args,metrics:Dict):
    path.parent.mkdir(parents=True,exist_ok=True)
    checkpoint={'epoch':epoch,'embedding_model_state_dict':embedding_model.state_dict(),'trm_model_state_dict':trm_model.state_dict(),'item_optimizer_state_dict':item_optimizer.state_dict(),'dense_optimizer_state_dict':dense_optimizer.state_dict(),'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},'metrics':metrics,'optimizer_scheme':{'item':'SGD_sparse_no_momentum','dense':'AdamW'}}
    torch.save(checkpoint,path)
    with open(path.with_suffix('.json'),'w') as f: json.dump({'epoch':epoch,'args':checkpoint['args'],'metrics':metrics,'optimizer_scheme':checkpoint['optimizer_scheme']},f,indent=2)
    return path

def load_checkpoint(path:Path,embedding_model:nn.Module,trm_model:nn.Module,item_optimizer:torch.optim.Optimizer,dense_optimizer:torch.optim.Optimizer,device:torch.device):
    if not path.exists(): raise FileNotFoundError(f'Checkpoint not found: {path}')
    print(f'\nLoading checkpoint: {path}')
    checkpoint=torch.load(path,map_location=device,weights_only=False)
    required={'embedding_model_state_dict','trm_model_state_dict','item_optimizer_state_dict','dense_optimizer_state_dict'}
    missing=required.difference(checkpoint.keys())
    if missing: raise RuntimeError('Checkpoint uses the old single-AdamW format and cannot be resumed by this trainer. Missing keys: '+', '.join(sorted(missing)))
    embedding_model.load_state_dict(checkpoint['embedding_model_state_dict']); trm_model.load_state_dict(checkpoint['trm_model_state_dict']); item_optimizer.load_state_dict(checkpoint['item_optimizer_state_dict']); dense_optimizer.load_state_dict(checkpoint['dense_optimizer_state_dict'])
    saved_epoch=int(checkpoint['epoch']); saved_metrics=checkpoint.get('metrics',{})
    print(f'  Restored epoch: {saved_epoch}'); print('  Restored embedding weights: ✓'); print('  Restored TRM-UI weights: ✓'); print('  Restored sparse item SGD: ✓'); print('  Restored dense AdamW: ✓')
    return saved_epoch,saved_metrics

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train TRM-UI on a controlled TAOBAO-MM subset."
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

    parser.add_argument("--history-length", type=int, default=DEFAULT_HISTORY_LENGTH)
    parser.add_argument("--train-samples", type=int, default=DEFAULT_TRAIN_SAMPLES)
    parser.add_argument("--val-samples", type=int, default=DEFAULT_VAL_SAMPLES)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)

    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--item-lr", type=float, default=1e-2, help="Learning rate for sparse item-embedding SGD.")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=DEFAULT_GRAD_CLIP,
    )

    parser.add_argument("--h-cycles", type=int, default=DEFAULT_H_CYCLES)
    parser.add_argument("--l-cycles", type=int, default=DEFAULT_L_CYCLES)
    parser.add_argument("--l-layers", type=int, default=DEFAULT_L_LAYERS)
    parser.add_argument("--hidden-size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--num-heads", type=int, default=DEFAULT_NUM_HEADS)

    parser.add_argument(
        "--item-embed-dim",
        type=int,
        default=DEFAULT_ITEM_EMBED_DIM,
    )
    parser.add_argument(
        "--category-embed-dim",
        type=int,
        default=DEFAULT_CATEGORY_EMBED_DIM,
    )

    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--log-every", type=int, default=50)

    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=CHECKPOINT_DIR,
    )

    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to a TRM-UI checkpoint to resume from.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)

    device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    print("=" * 70)
    print("TRM-UI controlled training experiment")
    print("=" * 70)
    print(f"Device:          {device}")
    if device.type == "cuda":
        print(f"GPU:             {torch.cuda.get_device_name(device)}")
    print(f"Train samples:   {args.train_samples}")
    print(f"Validation:      {args.val_samples}")
    print(f"Batch size:      {args.batch_size}")
    print(f"Epochs:          {args.epochs}")
    print(f"History length:  {args.history_length}")
    print(f"Dense LR:        {args.lr}")
    print(f"Item sparse LR:  {args.item_lr}")
    print(f"TRM hidden:      {args.hidden_size}")
    print(f"H cycles:        {args.h_cycles}")
    print(f"L cycles:        {args.l_cycles}")
    print(f"L layers:        {args.l_layers}")
    print("=" * 70)

    if not args.item_vocab.exists():
        raise FileNotFoundError(
            f"Missing item vocabulary: {args.item_vocab}"
        )

    if not args.category_vocab.exists():
        raise FileNotFoundError(
            f"Missing category vocabulary: {args.category_vocab}"
        )

    embedding_model, trm_model, item_mapper, category_mapper = build_models(
        args,
        device,
    )

    print("\nVocabulary:")
    print(f"  item rows:       {item_mapper.vocab_size}")
    print(f"  item UNK:        {item_mapper.unk_index}")
    print(f"  category rows:   {category_mapper.vocab_size}")
    print(f"  category UNK:    {category_mapper.unk_index}")

    embedding_params = sum(
        p.numel() for p in embedding_model.parameters()
    )
    trm_params = sum(
        p.numel() for p in trm_model.parameters()
    )

    print("\nParameters:")
    print(f"  embeddings:      {embedding_params:,}")
    print(f"  TRM-UI:          {trm_params:,}")
    print(f"  total:           {embedding_params + trm_params:,}")
    print(f"  item embedding sparse: {embedding_model.item_embedding.sparse}")

    item_optimizer = torch.optim.SGD(
        embedding_model.item_embedding.parameters(),
        lr=args.item_lr,
        momentum=0.0,
        weight_decay=0.0,
    )
    dense_optimizer = torch.optim.AdamW(
        list(embedding_model.category_embedding.parameters()) + list(trm_model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    # ---------------------------------------------------------------
    # Fixed vocabulary is reused for every split.
    # ---------------------------------------------------------------
    print("\nCreating training dataset...")
    train_dataset = get_taobao_dataset(
        split="train",
        history_length=args.history_length,
        item_vocab_path=str(args.item_vocab),
        category_vocab_path=str(args.category_vocab),
    )

    print("Creating validation dataset...")
    val_dataset = get_taobao_dataset(
        split="test",
        history_length=args.history_length,
        item_vocab_path=str(args.item_vocab),
        category_vocab_path=str(args.category_vocab),
    )

    # Resume from an existing checkpoint when requested.
    start_epoch = 1
    history = []

    if args.resume is not None:
        saved_epoch, saved_metrics = load_checkpoint(
            path=args.resume,
            embedding_model=embedding_model,
            trm_model=trm_model,
            item_optimizer=item_optimizer,
            dense_optimizer=dense_optimizer,
            device=device,
        )

        start_epoch = saved_epoch + 1

        if saved_metrics:
            history.append(
                {
                    "epoch": saved_epoch,
                    **saved_metrics,
                }
            )

        if start_epoch > args.epochs:
            print(
                f"Checkpoint is already at epoch {saved_epoch}, "
                f"while --epochs={args.epochs}. Nothing to train."
            )
            return

    print(f"\nStarting training from epoch {start_epoch}.")

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'=' * 70}")
        print(f"EPOCH {epoch}/{args.epochs}")
        print("=" * 70)

        train_metrics = train_one_epoch(
            train_dataset=train_dataset,
            embedding_model=embedding_model,
            trm_model=trm_model,
            item_optimizer=item_optimizer,
            dense_optimizer=dense_optimizer,
            device=device,
            args=args,
            epoch=epoch,
        )

        print(
            f"\nTraining result: "
            f"loss={train_metrics['loss']:.6f}, "
            f"samples={train_metrics['samples']}, "
            f"throughput={train_metrics['samples_per_second']:.2f} samples/s"
        )

        val_metrics = evaluate(
            val_dataset=val_dataset,
            embedding_model=embedding_model,
            trm_model=trm_model,
            device=device,
            args=args,
        )

        auc_text = (
            f"{val_metrics['auc']:.6f}"
            if math.isfinite(val_metrics["auc"])
            else "NaN"
        )

        print(
            f"Validation result: "
            f"loss={val_metrics['loss']:.6f}, "
            f"AUC={auc_text}, "
            f"samples={val_metrics['samples']}"
        )

        metrics = {
            "train": train_metrics,
            "validation": val_metrics,
        }

        history.append(
            {
                "epoch": epoch,
                **metrics,
            }
        )

        checkpoint_path = (
            args.checkpoint_dir
            / f"trm_ui_epoch_{epoch:03d}.pt"
        )

        save_checkpoint(
            path=checkpoint_path,
            epoch=epoch,
            embedding_model=embedding_model,
            trm_model=trm_model,
            item_optimizer=item_optimizer,
            dense_optimizer=dense_optimizer,
            args=args,
            metrics=metrics,
        )

        print(f"Checkpoint saved: {checkpoint_path}")

        # Save a compact training history after every epoch.
        history_path = args.checkpoint_dir / "training_history.json"

        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    for record in history:
        auc = record["validation"]["auc"]
        auc_text = f"{auc:.6f}" if math.isfinite(auc) else "NaN"

        print(
            f"Epoch {record['epoch']}: "
            f"train_loss={record['train']['loss']:.6f}, "
            f"val_loss={record['validation']['loss']:.6f}, "
            f"val_auc={auc_text}"
        )


if __name__ == "__main__":
    main()
