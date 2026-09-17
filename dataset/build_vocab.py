"""
Build a stable, bounded item/category vocabulary from TAOBAO-MM.

Vocabulary convention:
    0       = PAD
    1..N    = known IDs
    N + 1   = UNK

The persisted mapping is reused for training, validation, testing, and inference.
The item vocabulary can contain up to 10M known IDs without allocating a
10M-row tensor in this builder; only encountered IDs are stored.

For a controlled diagnostic, use --splits train,test. For a strict final
benchmark, use train only.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from dataset.taobao_dataset import get_taobao_dataset
from dataset.taobao_features import IDMapper, save_mapper

DEFAULT_MAX_ITEMS=10_000_000
DEFAULT_MAX_CATEGORIES=100_000
DEFAULT_NUM_SAMPLES=10_000
DEFAULT_HISTORY_LENGTH=50

def build_vocab(splits, num_samples, output_dir, history_length, max_items, max_categories):
    output_path=Path(output_dir); output_path.mkdir(parents=True, exist_ok=True)
    item_mapper=IDMapper(max_size=max_items); category_mapper=IDMapper(max_size=max_categories)
    print('='*70); print('TRM-UI vocabulary builder'); print('='*70)
    print(f"Splits:              {', '.join(splits)}")
    print(f"Samples per split:   {'all' if num_samples<=0 else f'{num_samples:,}'}")
    print(f"Max item IDs:        {max_items:,}")
    print(f"Max category IDs:    {max_categories:,}")
    print(f"History length:      {history_length}"); print('='*70)
    for split in splits:
        dataset=get_taobao_dataset(split=split, history_length=history_length)
        print(f"\nBuilding vocabulary from split={split!r}...")
        for i,sample in enumerate(dataset, start=1):
            for x in sample['history_items'].tolist(): item_mapper.add(x)
            for x in sample['history_categories'].tolist(): category_mapper.add(x)
            item_mapper.add(int(sample['target_item'].item()))
            category_mapper.add(int(sample['target_category'].item()))
            if i % 10000 == 0:
                print(f"  {split}: {i:,} samples | items={len(item_mapper):,}/{max_items:,} | categories={len(category_mapper):,}/{max_categories:,}")
            if num_samples>0 and i>=num_samples: break
            if len(item_mapper)>=max_items and len(category_mapper)>=max_categories:
                print('  Both vocabulary caps reached; stopping.'); break
    save_mapper(item_mapper, output_path/'item_vocab.json')
    save_mapper(category_mapper, output_path/'category_vocab.json')
    print('\nVocabulary complete.')
    print(f"Known items:       {len(item_mapper):,}")
    print(f"Item vocab rows:   {item_mapper.vocab_size:,}")
    print(f"Item UNK index:    {item_mapper.unk_index:,}")
    print(f"Known categories:  {len(category_mapper):,}")
    print(f"Category rows:     {category_mapper.vocab_size:,}")
    print(f"Category UNK:      {category_mapper.unk_index:,}")
    print(f"\nSaved to: {output_path}")

def main():
    p=argparse.ArgumentParser(description='Build a large stable TAOBAO-MM ID vocabulary.')
    p.add_argument('--splits', default='train', help='Comma-separated splits, e.g. train or train,test.')
    p.add_argument('--num-samples', type=int, default=DEFAULT_NUM_SAMPLES, help='Samples per split; 0 scans until dataset/caps.')
    p.add_argument('--history-length', type=int, default=DEFAULT_HISTORY_LENGTH)
    p.add_argument('--max-items', type=int, default=DEFAULT_MAX_ITEMS)
    p.add_argument('--max-categories', type=int, default=DEFAULT_MAX_CATEGORIES)
    p.add_argument('--output-dir', default='vocab/dev')
    a=p.parse_args()
    splits=[s.strip() for s in a.splits.split(',') if s.strip()]
    if not splits: raise ValueError('--splits must contain at least one split.')
    if a.max_items<1 or a.max_categories<1: raise ValueError('Vocabulary caps must be >= 1.')
    build_vocab(splits,a.num_samples,a.output_dir,a.history_length,a.max_items,a.max_categories)
if __name__=='__main__': main()
