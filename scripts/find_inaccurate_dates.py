"""Find out-of-season GBIF records likely caused by year-end date defaults.

When GBIF records are submitted with only a year known, the date is often
stamped as Dec 31 (or similarly coarse). Those records get binned to
``week_48`` by ``utils/gbifutils.py:date_to_week``, and feed summer-species
records into late-December training signal — which the model then amplifies
into spurious winter predictions (see DESCRIPTION_MEASSURES.md for context).

This script streams the large occurrence.txt from a GBIF Darwin-Core Archive,
keeps only records for user-specified migrant species that fall in Dec 22–31,
and prints a breakdown of publishers / datasets / dates responsible.

Usage::

    python scripts/find_inaccurate_dates.py \\
        --occurrence /media/pc/HD1/aibirder_model_data/gbif_norway_aves_mammalia_2000_202604/occurrence.txt

    # different species set
    python scripts/find_inaccurate_dates.py -i occurrence.txt --taxon_keys 9687165 2492956

    # a different "late-year" window
    python scripts/find_inaccurate_dates.py -i occurrence.txt --month 12 --day_from 22 --day_to 31
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


# Columns needed for filtering and breakdowns. We still load the *full*
# row (no ``usecols``) so the output dump contains every Darwin-Core field,
# but these are the ones we reference directly.
FILTER_COLS = [
    'taxonKey', 'year', 'month', 'day',
    'publisher', 'datasetName', 'datasetKey',
    'eventDate', 'verbatimEventDate',
]

# Default target species: migrants whose week-48 predictions are clearly wrong.
# GBIF taxonKey → common name. Seeded from the Oslo week-48 top-100 audit:
# species flagged ❌ (long-distance migrants wintering in sub-Saharan Africa)
# or ⚠️ (short-distance migrants rare in Norway in December).
DEFAULT_MIGRANTS = {
    # Long-distance (Afro-Palearctic) migrants — ❌
    5228676:  'Common Swift',
    11034996: 'Lesser Whitethroat',
    2493052:  'Willow Warbler',
    9367409:  'Common Tern',
    2492576:  'Spotted Flycatcher',
    2492606:  'European Pied Flycatcher',
    2492942:  'Garden Warbler',
    2481800:  'Common Sandpiper',
    9515886:  'Barn Swallow',
    10507373: 'Greater Whitethroat',
    2489214:  'Western House-Martin',
    8128385:  'Wood Warbler',
    5231240:  'Northern Wheatear',
    5739317:  'Common Redstart',
    # Short-distance migrants / rare-in-winter — ⚠️
    2481174:  'Lesser Black-backed Gull',
    9687165:  'White Wagtail',
    2492956:  'Eurasian Blackcap',
    2490310:  'Gray Wagtail',
    2493091:  'Common Chiffchaff',
    8000602:  'Eurasian Wigeon',
    7788295:  'Eurasian Oystercatcher',
    8214667:  'Green-winged Teal',
    5228199:  'Eurasian Moorhen',
    2482048:  'Little Grebe',
    2490266:  'Meadow Pipit',
    2498112:  'Northern Pintail',
    2491557:  'Reed Bunting',
    2498238:  'Velvet Scoter',
    # Partial-migrant baseline case — 🟡
    9616058:  'Eurasian Kestrel',
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--occurrence', '-i', required=True,
                        help='Path to occurrence.txt from the GBIF Darwin-Core Archive')
    parser.add_argument('--taxon_keys', type=int, nargs='+', default=None,
                        help=f'GBIF taxonKeys to check (default: {sorted(DEFAULT_MIGRANTS)})')
    parser.add_argument('--month', type=int, default=12,
                        help='Month to filter (default: 12 = December)')
    parser.add_argument('--day_from', type=int, default=22,
                        help='Start day inclusive (default: 22 — week 48 in 48-week scheme)')
    parser.add_argument('--day_to', type=int, default=31,
                        help='End day inclusive (default: 31)')
    parser.add_argument('--chunksize', type=int, default=50_000,
                        help='Rows per chunk (default: 50000; lower than before '
                             'because we now read every column in the DwC-A)')
    parser.add_argument('--top', type=int, default=30,
                        help='Number of top source rows to show (default: 30)')
    parser.add_argument('--output', '-o', default=None,
                        help='Optional CSV path to dump all matched records '
                             '(includes every column from occurrence.txt)')
    args = parser.parse_args()

    targets = set(args.taxon_keys) if args.taxon_keys else set(DEFAULT_MIGRANTS)
    targets_str = {str(t) for t in targets}
    names = {str(tk): DEFAULT_MIGRANTS.get(tk, str(tk)) for tk in targets}

    occ = Path(args.occurrence)
    if not occ.is_file():
        sys.exit(f'error: occurrence file not found: {occ}')

    print(f'Streaming {occ} ({occ.stat().st_size / 1e9:.1f} GB) in '
          f'{args.chunksize:,}-row chunks …', flush=True)
    print(f'Filter: taxonKey ∈ {sorted(targets)},  month={args.month},  '
          f'day ∈ [{args.day_from}, {args.day_to}]', flush=True)

    # Load *every* column so matched rows can be dumped in full. Only the
    # FILTER_COLS are referenced during filtering.
    reader = pd.read_csv(
        occ, sep='\t',
        chunksize=args.chunksize, dtype=str,
        on_bad_lines='skip', quoting=3,  # csv.QUOTE_NONE
        low_memory=True,
    )

    pieces = []
    month_str = str(args.month)
    for i, chunk in enumerate(reader):
        mask = chunk['taxonKey'].isin(targets_str) & (chunk['month'] == month_str)
        if not mask.any():
            if (i + 1) % 20 == 0:
                print(f'  processed {(i + 1) * args.chunksize:>12,} rows, '
                      f'matches so far: {sum(len(p) for p in pieces):,}', flush=True)
            continue
        sub = chunk.loc[mask].copy()
        day = pd.to_numeric(sub['day'], errors='coerce')
        sub = sub[day.between(args.day_from, args.day_to)]
        if len(sub):
            pieces.append(sub)
        if (i + 1) % 20 == 0:
            print(f'  processed {(i + 1) * args.chunksize:>12,} rows, '
                  f'matches so far: {sum(len(p) for p in pieces):,}', flush=True)

    df = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=FILTER_COLS)
    print(f'\nKept {len(df):,} records matching the filter.\n')

    if df.empty:
        return

    # --- Breakdowns ---
    df['species'] = df['taxonKey'].map(names).fillna(df['taxonKey'])

    print(f'── Records per species ──')
    print(df['species'].value_counts().to_string())

    print(f'\n── Day-of-month distribution (all species combined) ──')
    day_counts = pd.to_numeric(df['day'], errors='coerce').value_counts().sort_index()
    for d, n in day_counts.items():
        bar = '█' * int(40 * n / day_counts.max())
        print(f'  day {int(d):>2}  {int(n):>6}  {bar}')

    print(f'\n── Top {args.top} sources (publisher / datasetName / species) ──')
    grp = (
        df.groupby(['publisher', 'datasetName', 'species'], dropna=False)
        .size().reset_index(name='count')
        .sort_values('count', ascending=False)
        .head(args.top)
    )
    with pd.option_context('display.max_colwidth', 60, 'display.width', 200):
        print(grp.to_string(index=False))

    print(f'\n── Top {args.top} datasets by count (datasetKey + total) ──')
    key_grp = df.groupby('datasetKey').size().sort_values(ascending=False).head(args.top)
    for key, n in key_grp.items():
        print(f'  {n:>6}  {key}')

    print(f'\n── Sample verbatim date strings (reveal "year-only" patterns) ──')
    verbatim_counts = df['verbatimEventDate'].fillna('(none)').value_counts().head(20)
    for v, n in verbatim_counts.items():
        print(f'  {n:>6}  {v!r}')

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        # Drop the derived 'species' helper column so the dump contains
        # only the original Darwin-Core fields. Keep tab-separated to match
        # the source occurrence.txt; callers can re-import it identically.
        dump = df.drop(columns=['species'], errors='ignore')
        dump.to_csv(out, index=False, sep='\t')
        print(f'\nWrote all {len(df):,} matched records to {out}')


if __name__ == '__main__':
    main()
