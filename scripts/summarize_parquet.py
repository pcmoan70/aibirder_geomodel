"""Summarize a (potentially huge) Parquet file.

Prints:
  1. File size, row count, schema (column names + arrow types).
  2. Per-column stats (nulls, numeric min/max/mean, unique-ish sample for
     object columns). Skip with ``--no-stats``.
  3. Head rows (default 20).
  4. Tail rows (default 10).
  5. Random-sample rows (default 0 — pass ``--sample N`` to enable).

Efficient for large files: the schema + row count come from parquet
metadata (no data read). Data reads honour ``--columns`` so you only load
what you need; row selection uses positional iloc on the resulting
DataFrame (requires the selected columns to fit in RAM).

Usage::

    python scripts/summarize_parquet.py -i data.parquet
    python scripts/summarize_parquet.py -i data.parquet --head 0 --tail 50
    python scripts/summarize_parquet.py -i data.parquet --sample 30
    python scripts/summarize_parquet.py -i data.parquet --columns lat,lon,week_17
    python scripts/summarize_parquet.py -i data.parquet --columns 0:10
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def _human_bytes(n: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if n < 1024:
            return f'{n:.1f} {unit}'
        n /= 1024
    return f'{n:.1f} PB'


def _resolve_columns(spec: Optional[str], all_columns: Sequence[str]) -> Optional[List[str]]:
    """Parse a --columns argument.

    Accepts:
      - comma-separated names: ``lat,lon,week_17``
      - integer index slice ``start:end`` (Python-style, end exclusive)
      - mix: ``lat,5:10,week_17``
    Returns None if *spec* is None (load all columns).
    """
    if spec is None:
        return None
    picked: List[str] = []
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if ':' in part and all(c.isdigit() or c in ':' for c in part):
            start_s, _, end_s = part.partition(':')
            start = int(start_s) if start_s else 0
            end = int(end_s) if end_s else len(all_columns)
            picked.extend(all_columns[start:end])
        else:
            if part not in all_columns:
                print(f'warning: column {part!r} not in file — skipped', file=sys.stderr)
                continue
            picked.append(part)
    # Preserve order and de-duplicate.
    seen = set()
    out = []
    for c in picked:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _print_schema(pf: pq.ParquetFile) -> None:
    schema = pf.schema_arrow
    print(f'\nSchema ({len(schema.names)} columns):')
    print(f'  {"#":>3}  {"name":<30}  {"arrow_type":<20}')
    print(f'  {"-"*3}  {"-"*30}  {"-"*20}')
    for i, name in enumerate(schema.names):
        atype = schema.field(i).type
        print(f'  {i:>3}  {name:<30}  {str(atype):<20}')


def _print_stats(df: pd.DataFrame) -> None:
    """Per-column stats summary."""
    print(f'\nPer-column stats (over {len(df):,} loaded rows):')
    print(f'  {"name":<30}  {"dtype":<15}  {"nulls":>8}  {"summary"}')
    print(f'  {"-"*30}  {"-"*15}  {"-"*8}  {"-"*40}')
    for col in df.columns:
        s = df[col]
        n_null = int(s.isna().sum())
        dt = str(s.dtype)
        if pd.api.types.is_numeric_dtype(s.dtype):
            nn = s.dropna()
            if nn.size:
                summary = f'min={nn.min():.3g}  mean={nn.mean():.3g}  max={nn.max():.3g}'
            else:
                summary = '(all null)'
        elif pd.api.types.is_bool_dtype(s.dtype):
            summary = f'true={int(s.sum())}  false={int((~s.fillna(False)).sum())}'
        else:
            # object / list / string — count unique, show a short sample
            try:
                uniq = s.dropna().astype(str).nunique(dropna=True)
                sample = s.dropna().astype(str).head(1).iloc[0] if n_null < len(s) else ''
                if len(sample) > 40:
                    sample = sample[:37] + '...'
                summary = f'unique≈{uniq}  sample={sample!r}'
            except Exception as exc:  # non-stringifiable (e.g. shapely)
                summary = f'(non-str dtype: {exc.__class__.__name__})'
        print(f'  {col:<30}  {dt:<15}  {n_null:>8}  {summary}')


def _print_slice(df: pd.DataFrame, start: int, stop: int, label: str) -> None:
    """Print a positional slice of the DataFrame."""
    if start >= stop or start >= len(df):
        return
    sub = df.iloc[start:stop]
    print(f'\n{label} (rows {start}..{stop - 1}):')
    with pd.option_context('display.max_columns', None,
                           'display.width', 200,
                           'display.max_colwidth', 60):
        print(sub.to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--input', '-i', required=True, help='Path to parquet file')
    parser.add_argument('--head', type=int, default=20, help='Rows to show from the top (default 20)')
    parser.add_argument('--tail', type=int, default=10, help='Rows to show from the bottom (default 10)')
    parser.add_argument('--sample', type=int, default=0, help='Random rows to show (default 0)')
    parser.add_argument('--seed', type=int, default=0, help='Seed for --sample (default 0)')
    parser.add_argument('--columns', default=None,
                        help='Column selection: comma-separated names, "start:end" '
                             'positional slice, or mix (e.g. "lat,5:10,week_17")')
    parser.add_argument('--no-stats', action='store_true', help='Skip per-column stats')
    args = parser.parse_args()

    inp = Path(args.input)
    if not inp.is_file():
        sys.exit(f'error: not a file: {inp}')

    pf = pq.ParquetFile(inp)
    n_rows = pf.metadata.num_rows
    n_cols = len(pf.schema_arrow.names)
    print(f'File:     {inp}')
    print(f'Size:     {_human_bytes(inp.stat().st_size)}')
    print(f'Rows:     {n_rows:,}')
    print(f'Columns:  {n_cols}')
    print(f'Row groups: {pf.num_row_groups}')

    _print_schema(pf)

    cols = _resolve_columns(args.columns, list(pf.schema_arrow.names))
    if cols is not None:
        print(f'\nLoading {len(cols)} selected columns ...', file=sys.stderr)
    else:
        print(f'\nLoading all {n_cols} columns ...', file=sys.stderr)

    df = pd.read_parquet(inp, columns=cols)
    print(f'Loaded shape: {df.shape}', file=sys.stderr)

    if not args.no_stats:
        _print_stats(df)

    if args.head > 0:
        _print_slice(df, 0, min(args.head, len(df)), f'Head — first {args.head}')

    if args.sample > 0 and len(df) > 0:
        n = min(args.sample, len(df))
        rng = np.random.default_rng(args.seed)
        idx = np.sort(rng.choice(len(df), size=n, replace=False))
        sub = df.iloc[idx]
        print(f'\nRandom sample ({n} rows, seed={args.seed}):')
        with pd.option_context('display.max_columns', None,
                               'display.width', 200,
                               'display.max_colwidth', 60):
            print(sub.to_string())

    if args.tail > 0:
        start = max(0, len(df) - args.tail)
        _print_slice(df, start, len(df), f'Tail — last {args.tail}')


if __name__ == '__main__':
    main()
