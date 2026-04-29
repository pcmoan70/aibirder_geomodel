"""Inventory GBIF DwC-A zips + FinBIF csv.gz archives in a folder.

For every archive, produces per-year-per-class counts of observations,
unique locations (rounded coordinates) and unique species. All files are
streamed in chunks so the script handles tens-of-GB inputs comfortably.

Output (CSV by default, also pretty-printed to stdout):

    archive, year, class, n_obs, n_locations, n_species

Usage::

    python scripts/inventory_archives.py \\
        --dir /media/pc/HD1/aibirder_model_data/gbif_archives \\
        --output /tmp/archives_inventory.csv

    # Restrict to specific archives, big chunks, speed mode
    python scripts/inventory_archives.py \\
        --dir /media/pc/HD1/aibirder_model_data/gbif_archives \\
        --pattern '*.csv.gz' --chunksize 1000000

The set-based unique counts use bounded memory because they're per
(archive, year, class) grouping — typical Nordic archive has 0–600
species and a few 100k unique 5km locations per (year, class), which
fits in well under 200 MB total RAM.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Set

import pandas as pd


# Columns we care about, mapped per archive flavour.
GBIF_COLS = ['year', 'class', 'taxonKey', 'verbatimScientificName',
             'decimalLatitude', 'decimalLongitude']
FINBIF_COLS = ['eventDate', 'class', 'taxonKey', 'verbatimScientificName',
               'latitude', 'longitude']


def _open_gbif_occurrence(zip_path: Path):
    """Return (open_file_obj, file_size) for ``occurrence.txt`` inside *zip_path*."""
    z = zipfile.ZipFile(zip_path, 'r')
    name = next(
        (n for n in z.namelist()
         if n == 'occurrence.txt' or n.endswith('/occurrence.txt')),
        None,
    )
    if name is None:
        z.close()
        raise FileNotFoundError(f'no occurrence.txt in {zip_path}')
    info = z.getinfo(name)
    f = z.open(name)
    return f, info.file_size, z


def _iter_gbif_chunks(zip_path: Path, chunksize: int) -> Iterable[pd.DataFrame]:
    """Yield DataFrames with normalised column names from a GBIF zip."""
    f, _size, z = _open_gbif_occurrence(zip_path)
    try:
        reader = pd.read_csv(
            f, sep='\t', usecols=lambda c: c in GBIF_COLS,
            chunksize=chunksize, dtype=str,
            on_bad_lines='skip', quoting=csv.QUOTE_NONE,
            low_memory=True, encoding='utf-8', encoding_errors='replace',
        )
        for chunk in reader:
            chunk = chunk.rename(columns={
                'decimalLatitude': 'lat',
                'decimalLongitude': 'lon',
                'verbatimScientificName': 'sci',
            })
            chunk['year'] = pd.to_numeric(chunk['year'], errors='coerce')
            chunk['class'] = chunk['class'].str.lower().fillna('')
            yield chunk
    finally:
        f.close()
        z.close()


def _iter_finbif_chunks(gz_path: Path, chunksize: int) -> Iterable[pd.DataFrame]:
    """Yield DataFrames with normalised column names from a FinBIF gzipped CSV."""
    reader = pd.read_csv(
        gz_path, compression='gzip',
        usecols=lambda c: c in FINBIF_COLS,
        chunksize=chunksize, dtype=str,
        on_bad_lines='skip', low_memory=True,
        encoding='utf-8', encoding_errors='replace',
    )
    for chunk in reader:
        chunk = chunk.rename(columns={
            'latitude': 'lat',
            'longitude': 'lon',
            'verbatimScientificName': 'sci',
        })
        # Parse year from eventDate's first 4 chars (faster than full date parse).
        if 'eventDate' in chunk.columns:
            chunk['year'] = pd.to_numeric(
                chunk['eventDate'].str[:4], errors='coerce',
            )
        else:
            chunk['year'] = pd.NA
        if 'class' in chunk.columns:
            chunk['class'] = chunk['class'].str.lower().fillna('')
        else:
            chunk['class'] = ''
        yield chunk


def _process_archive(
    path: Path,
    chunksize: int,
    coord_round: int,
) -> Dict[Tuple[int, str], Dict]:
    """Return ``{(year, class): {n_obs, locations, species}}`` for *path*."""
    name = path.name
    if path.suffix == '.zip':
        chunks = _iter_gbif_chunks(path, chunksize)
        kind = 'gbif'
    elif name.endswith('.csv.gz'):
        chunks = _iter_finbif_chunks(path, chunksize)
        kind = 'finbif'
    else:
        print(f'  [skip] {name}: unknown format')
        return {}

    print(f'  [{kind}] streaming {name} (chunksize={chunksize:,}) ...', flush=True)
    by_key: Dict[Tuple[int, str], Dict] = defaultdict(
        lambda: {'n_obs': 0, 'locs': set(), 'sps': set()}
    )
    n_total = 0
    for chunk in chunks:
        # Filter rows with usable coords
        chunk = chunk.dropna(subset=['lat', 'lon', 'year'])
        if chunk.empty:
            continue
        # Bin coords to limit set-size of locations
        try:
            lat = pd.to_numeric(chunk['lat'], errors='coerce').round(coord_round)
            lon = pd.to_numeric(chunk['lon'], errors='coerce').round(coord_round)
        except Exception:
            continue
        chunk['year'] = chunk['year'].astype('Int64')
        chunk['_loc'] = lat.astype(str) + ',' + lon.astype(str)
        # Use sci_name (or taxonKey if missing) as species identifier
        species_key = chunk['sci'].fillna('').replace('', None)
        species_key = species_key.fillna(chunk.get('taxonKey', ''))
        chunk['_sp'] = species_key

        n_total += len(chunk)
        # Group + accumulate
        for (yr, cls), grp in chunk.groupby(['year', 'class']):
            key = (int(yr), cls)
            entry = by_key[key]
            entry['n_obs'] += len(grp)
            entry['locs'].update(grp['_loc'].tolist())
            entry['sps'].update(grp['_sp'].dropna().tolist())
        if n_total % (chunksize * 5) == 0:
            print(f'    rows seen: {n_total:,}', flush=True)
    return by_key


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--dir', required=True,
                        help='Folder containing GBIF zips and FinBIF csv.gz files.')
    parser.add_argument('--output', '-o', default=None,
                        help='Output CSV path. Default: <dir>/inventory.csv')
    parser.add_argument('--chunksize', type=int, default=200_000)
    parser.add_argument('--pattern', default='*',
                        help='Glob filter on filenames (default: all).')
    parser.add_argument('--coord-round', type=int, default=2,
                        help='Decimal places to round lat/lon to before counting '
                             'unique locations (default 2 ≈ 1 km bins).')
    args = parser.parse_args()

    folder = Path(args.dir)
    if not folder.is_dir():
        sys.exit(f'error: not a directory: {folder}')
    out = Path(args.output) if args.output else folder / 'inventory.csv'

    pattern = args.pattern
    files = sorted([
        p for p in folder.glob(pattern)
        if p.is_file() and (p.suffix == '.zip' or p.name.endswith('.csv.gz'))
    ])
    if not files:
        sys.exit(f'No matching archives in {folder} (pattern={pattern!r}).')
    print(f'Found {len(files)} archive(s) under {folder}')
    for p in files:
        print(f'  {p.name}  ({p.stat().st_size/1e9:.2f} GB)')
    print()

    rows = []
    for path in files:
        print(f'=== {path.name} ===')
        try:
            stats = _process_archive(path, args.chunksize, args.coord_round)
        except Exception as exc:
            print(f'  ERROR: {exc.__class__.__name__}: {exc}')
            continue
        for (year, cls), v in sorted(stats.items()):
            rows.append({
                'archive':     path.name,
                'year':        year,
                'class':       cls or '(unknown)',
                'n_obs':       v['n_obs'],
                'n_locations': len(v['locs']),
                'n_species':   len(v['sps']),
            })

    df = pd.DataFrame(rows)
    if df.empty:
        print('No data extracted — output file not written.')
        return
    df = df.sort_values(['archive', 'year', 'class']).reset_index(drop=True)
    df.to_csv(out, index=False)
    print(f'\nWrote {len(df)} rows to {out}')

    # --- additional summary CSVs ----------------------------------------
    # All summary files share the same column types so they parse uniformly
    # in pandas / spreadsheet tools.

    base = out.with_suffix('')

    # 1. Per-archive totals.
    by_arc = (
        df.groupby('archive', as_index=False)
          .agg(year_min=('year', 'min'),
               year_max=('year', 'max'),
               classes =('class', lambda s: ';'.join(sorted(s.unique()))),
               n_obs       =('n_obs', 'sum'),
               n_locations =('n_locations', 'sum'),
               n_species   =('n_species', 'sum'))
          .sort_values('archive')
    )
    arc_path = Path(f'{base}_by_archive.csv')
    by_arc.to_csv(arc_path, index=False)
    print(f'Wrote {len(by_arc)} rows to {arc_path}')

    # 2. Per (year, class) totals across all archives.
    by_yc = (
        df.groupby(['year', 'class'], as_index=False)
          .agg(n_obs       =('n_obs', 'sum'),
               n_locations =('n_locations', 'sum'),
               n_species   =('n_species', 'sum'),
               n_archives  =('archive', 'nunique'))
          .sort_values(['year', 'class'])
    )
    yc_path = Path(f'{base}_by_year_class.csv')
    by_yc.to_csv(yc_path, index=False)
    print(f'Wrote {len(by_yc)} rows to {yc_path}')

    # 3. Per-year totals (collapsed over class) across all archives.
    by_year = (
        df.groupby('year', as_index=False)
          .agg(n_obs       =('n_obs', 'sum'),
               n_locations =('n_locations', 'sum'),
               n_species   =('n_species', 'sum'),
               n_archives  =('archive', 'nunique'),
               classes     =('class', lambda s: ';'.join(sorted(s.unique()))))
          .sort_values('year')
    )
    year_path = Path(f'{base}_by_year.csv')
    by_year.to_csv(year_path, index=False)
    print(f'Wrote {len(by_year)} rows to {year_path}')

    print(
        '\nNote: n_locations and n_species in the aggregated summaries are '
        'sums of per-archive uniques, so they overcount any species / '
        'location bin that appears in more than one archive. Use the '
        'per-archive-per-class detail CSV for exact uniques.'
    )


if __name__ == '__main__':
    main()
