"""List taxon codes and species names in the current model vocabulary.

Reads the species vocab from a checkpoint (authoritative — not affected by
``labels.txt`` being overwritten by a later training run) and joins it with a
taxonomy CSV to produce human-readable names.

Usage::

    python scripts/list_species.py
    python scripts/list_species.py --checkpoint checkpoints/checkpoint_latest.pt
    python scripts/list_species.py --format csv --output species.csv
    python scripts/list_species.py --grep swallow
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch


def load_taxonomy(path: Path) -> Dict[str, Tuple[str, str]]:
    """Return ``{taxonKey: (sci_name, com_name)}``."""
    lookup: Dict[str, Tuple[str, str]] = {}
    if not path.exists():
        return lookup
    with path.open(encoding='utf-8') as f:
        reader = csv.DictReader(f)
        key_col = 'taxonKey' if 'taxonKey' in reader.fieldnames else 'species_code'
        for row in reader:
            k = str(row.get(key_col, '')).strip()
            if not k:
                continue
            lookup[k] = (row.get('sci_name', '') or '', row.get('com_name', '') or '')
    return lookup


def build_vocab(checkpoint_path: Path) -> Dict[int, str]:
    """Load ``idx_to_species`` (idx → taxonKey as string) from the checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    idx_to_species = ckpt.get('species_vocab', {}).get('idx_to_species', {})
    if not idx_to_species:
        raise RuntimeError(f'No species_vocab found in {checkpoint_path}')
    # Values can be numpy.int64; normalize to plain strings.
    return {int(k): str(int(v)) for k, v in idx_to_species.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--checkpoint', '-c', default='checkpoints/checkpoint_best.pt',
                        help='Checkpoint file (default: checkpoint_best.pt)')
    parser.add_argument('--taxonomy', '-t',
                        default='/media/pc/HD1/aibirder_model_data/combined_taxonomy.csv',
                        help='Taxonomy CSV with taxonKey/sci_name/com_name columns')
    parser.add_argument('--format', choices=['text', 'tsv', 'csv'], default='text',
                        help='Output format (default: text columns)')
    parser.add_argument('--output', '-o', default=None,
                        help='Write to file instead of stdout')
    parser.add_argument('--grep', default=None,
                        help='Case-insensitive filter on sci_name or com_name')
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        sys.exit(f'Checkpoint not found: {ckpt_path}')

    vocab = build_vocab(ckpt_path)
    tax = load_taxonomy(Path(args.taxonomy))

    rows = []
    q = args.grep.lower() if args.grep else None
    for idx in sorted(vocab):
        key = vocab[idx]
        sci, com = tax.get(key, ('', ''))
        if q is not None and q not in sci.lower() and q not in com.lower():
            continue
        rows.append((idx, key, sci, com))

    out = open(args.output, 'w', encoding='utf-8') if args.output else sys.stdout
    try:
        if args.format == 'text':
            out.write(f'{"idx":>4}  {"taxonKey":>10}  {"scientific":<35}  common\n')
            out.write('-' * 80 + '\n')
            for idx, key, sci, com in rows:
                out.write(f'{idx:>4}  {key:>10}  {sci:<35}  {com}\n')
        else:
            sep = '\t' if args.format == 'tsv' else ','
            writer = csv.writer(out, delimiter=sep)
            writer.writerow(['idx', 'taxonKey', 'sci_name', 'com_name'])
            writer.writerows(rows)
    finally:
        if args.output:
            out.close()

    n_shown = len(rows)
    n_total = len(vocab)
    msg = f'{n_shown}/{n_total} species'
    if q:
        msg += f" matching '{args.grep}'"
    if args.output:
        msg += f' → {args.output}'
    print(msg, file=sys.stderr)


if __name__ == '__main__':
    main()
