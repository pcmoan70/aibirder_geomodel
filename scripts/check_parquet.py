"""Check a (possibly huge) parquet file for read errors and optionally repair.

The BirdNET combined parquet has list columns (week_1..week_48) and can
trigger cross-version pyarrow bugs like ``Repetition level histogram size
mismatch``. This script:

  1. Reports pyarrow version, file size, row count, row group count.
  2. Attempts a straight ``pq.read_table`` to reproduce a read error.
  3. Falls back to per-row-group streaming and per-column reads to isolate
     which row groups / columns (if any) are actually broken.
  4. When ``--repair OUT`` is given, streams the file through ``iter_batches``
     and re-writes it to ``OUT`` — this drops stale column statistics that
     trigger the bug on older readers while preserving all data.

Usage::

    python scripts/check_parquet.py -i combined.parquet
    python scripts/check_parquet.py -i combined.parquet --repair combined_fixed.parquet
    python scripts/check_parquet.py -i combined.parquet --row-group-limit 5
"""

import argparse
import sys
import traceback
from pathlib import Path
from typing import List, Optional

import pyarrow as pa
import pyarrow.parquet as pq


def _human_bytes(n: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if n < 1024:
            return f'{n:.1f} {unit}'
        n /= 1024
    return f'{n:.1f} PB'


def _try_full_read(path: Path) -> Optional[str]:
    """Attempt a full read; return error message or None on success."""
    try:
        pq.read_table(path)
        return None
    except Exception as exc:
        return f'{type(exc).__name__}: {exc}'


def _diagnose_per_row_group(pf: pq.ParquetFile, limit: int) -> List[str]:
    """Try reading each row group one at a time, return list of errors."""
    errs: List[str] = []
    n = min(limit if limit > 0 else pf.num_row_groups, pf.num_row_groups)
    for rg in range(n):
        try:
            pf.read_row_group(rg)
        except Exception as exc:
            errs.append(f'  row_group {rg}: {type(exc).__name__}: {exc}')
    return errs


def _diagnose_per_column(pf: pq.ParquetFile) -> List[str]:
    """Try reading each column in isolation; reports which one(s) break."""
    errs: List[str] = []
    for name in pf.schema_arrow.names:
        try:
            pq.read_table(pf.metadata.metadata and pf.reader or pf, columns=[name])
        except Exception:
            try:
                pf.read(columns=[name])
            except Exception as exc:
                errs.append(f'  column {name!r}: {type(exc).__name__}: {exc}')
    return errs


def _repair(path: Path, out: Path, batch_size: int = 50_000) -> None:
    """Stream the input through iter_batches and write a fresh parquet.

    This works around cross-version stats bugs by dropping the old
    column/page statistics and letting the writer rebuild them from the
    actual data.
    """
    pf = pq.ParquetFile(path)
    schema = pf.schema_arrow
    out.parent.mkdir(parents=True, exist_ok=True)

    # Preserve the parquet key/value metadata where possible (e.g. GeoParquet
    # 'geo' key written by GeoPandas).
    kv_meta = pf.metadata.metadata or {}

    print(f'  streaming {pf.num_row_groups} row group(s) → {out}', flush=True)
    n_rows = 0
    writer = pq.ParquetWriter(
        out, schema=schema,
        compression='snappy',
        use_dictionary=True,
        write_statistics=True,
    )
    try:
        for batch in pf.iter_batches(batch_size=batch_size):
            tbl = pa.Table.from_batches([batch], schema=schema)
            writer.write_table(tbl)
            n_rows += batch.num_rows
            if n_rows % (batch_size * 10) == 0:
                print(f'    wrote {n_rows:,} rows ...', flush=True)
    finally:
        writer.close()

    # Re-attach the original key/value metadata (geo etc.) if present.
    if kv_meta:
        try:
            table = pq.read_table(out)
            table = table.replace_schema_metadata({**(table.schema.metadata or {}), **kv_meta})
            pq.write_table(table, out, compression='snappy')
        except Exception as exc:
            print(f'  warning: failed to restore key/value metadata: {exc}')

    print(f'  repaired: wrote {n_rows:,} rows to {out} '
          f'({_human_bytes(out.stat().st_size)})')


def _verify(path: Path) -> None:
    """Read the repaired file to confirm it round-trips cleanly."""
    print(f'\nVerifying {path} ...', flush=True)
    err = _try_full_read(path)
    if err is None:
        print(f'  OK — reads cleanly via pyarrow {pa.__version__}.')
    else:
        print(f'  STILL BROKEN: {err}')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--input', '-i', required=True, help='Parquet file to check')
    parser.add_argument('--repair', '-o', default=None,
                        help='Output path; when set, re-write the file via '
                             'iter_batches to fix cross-version stats bugs.')
    parser.add_argument('--batch-size', type=int, default=50_000,
                        help='Batch size for streaming repair (default 50000)')
    parser.add_argument('--row-group-limit', type=int, default=0,
                        help='Stop per-row-group diagnosis after N groups '
                             '(0 = check all). Useful on large files.')
    parser.add_argument('--per-column', action='store_true',
                        help='Also try per-column reads; slow for wide schemas.')
    args = parser.parse_args()

    inp = Path(args.input)
    if not inp.is_file():
        sys.exit(f'error: not a file: {inp}')

    print(f'pyarrow version: {pa.__version__}')
    print(f'file:            {inp}')
    print(f'size:            {_human_bytes(inp.stat().st_size)}')

    pf = pq.ParquetFile(inp)
    md = pf.metadata
    print(f'rows:            {md.num_rows:,}')
    print(f'columns:         {md.num_columns}')
    print(f'row_groups:      {pf.num_row_groups}')
    kv = md.metadata or {}
    if kv:
        keys = ', '.join(k.decode(errors='replace') for k in kv)
        print(f'key/value keys:  {keys}')

    print('\n--- Full-read attempt ---')
    err = _try_full_read(inp)
    if err is None:
        print('OK — pq.read_table() succeeded.')
    else:
        print(f'FAILED: {err}')

    if err is not None or args.per_column or args.row_group_limit:
        print('\n--- Per-row-group diagnosis ---')
        rg_errs = _diagnose_per_row_group(pf, args.row_group_limit)
        if rg_errs:
            print(f'{len(rg_errs)} row group(s) failed:')
            for e in rg_errs:
                print(e)
        else:
            print('All row groups readable individually.')

        if args.per_column:
            print('\n--- Per-column diagnosis ---')
            col_errs = _diagnose_per_column(pf)
            if col_errs:
                print(f'{len(col_errs)} column(s) failed:')
                for e in col_errs:
                    print(e)
            else:
                print('All columns readable individually.')

    if args.repair:
        out = Path(args.repair)
        print(f'\n--- Repair to {out} ---')
        try:
            _repair(inp, out, batch_size=args.batch_size)
            _verify(out)
        except Exception as exc:
            print(f'repair failed: {type(exc).__name__}: {exc}')
            traceback.print_exc()
            sys.exit(2)


if __name__ == '__main__':
    main()
