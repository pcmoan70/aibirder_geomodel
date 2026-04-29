"""Download Finnish observation data from FinBIF (laji.fi).

The Finnish national portal hosts vastly more bird records than what's
mirrored to GBIF. This script pulls observations directly from the
FinBIF warehouse API and emits a CSV with columns compatible with the
rest of our pipeline (``utils/gbifutils.py`` output schema), so the
result can be passed straight into ``utils/combine.py`` alongside the
GBIF-derived CSVs.

Auth
----
Get an access token by emailing the ``/v0/api-users`` endpoint::

    POST https://api.laji.fi/v0/api-users   data={'email': '<you>'}

A token will arrive in your inbox. Pass via ``--access-token`` or set
``FINBIF_TOKEN`` in the environment. All requests use the
``Authorization: Bearer <token>`` header.

Usage
-----
::

    export FINBIF_TOKEN=...
    python utils/finbif_download.py --classes Aves --year-from 2010 \\
        --output /media/pc/HD1/aibirder_model_data/finbif_aves_2010-2026.csv.gz

    # birds + mammals
    python utils/finbif_download.py --classes Aves Mammalia --year-from 2010 \\
        --output /tmp/finbif_aves_mammalia.csv.gz

Notes
-----
- FinBIF uses *informal taxon groups* (``MVL.1`` = Birds, ``MVL.2`` =
  Mammals) for class filtering. ``--classes Aves Mammalia`` is mapped to
  the right group IDs internally.
- Date filter uses ``time=YYYY/YYYY`` (FinBIF's syntax).
- ``--max-accuracy`` becomes the FinBIF ``coordinateAccuracyMax``
  parameter — drops imprecise records before they're even fetched.
- The output schema matches ``utils/gbifutils.py``'s ``OUTPUT_COLUMNS``
  so downstream processing requires no special-casing.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import gzip
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests

API_BASE = 'https://api.laji.fi/v0'

# Class name → FinBIF informal taxon group id.
# Full list at /v0/informal-taxon-groups; only the taxa we use here.
CLASS_TO_GROUP = {
    'aves':             'MVL.1',
    'mammalia':         'MVL.2',
    # Plants split in FinBIF: vascular plants and mosses are separate groups.
    'vascular_plants':  'MVL.343',
    'plants':           'MVL.343',     # alias for the common case
    'mosses':           'MVL.561',
    # MVL.233 is the umbrella for all fungi *and* lichens — what the user
    # almost always wants when asking for "fungi/lichen".
    'fungi':            'MVL.233',
    'fungi_lichens':    'MVL.233',
    'lichens':          'MVL.25',      # specific subset of fungi
}


# Where in the JSON each output column lives. First non-empty match wins.
FIELD_MAP = {
    'latitude':              ['gathering.conversions.wgs84CenterPoint.lat',
                              'gathering.conversions.wgs84.lat'],
    'longitude':             ['gathering.conversions.wgs84CenterPoint.lon',
                              'gathering.conversions.wgs84.lon'],
    'taxonKey':              ['unit.linkings.taxon.qname',
                              'unit.linkings.taxon.id'],
    'verbatimScientificName':['unit.linkings.taxon.scientificName',
                              'unit.taxonVerbatim'],
    'commonName':            ['unit.linkings.taxon.vernacularName.en',
                              'unit.linkings.taxon.vernacularName.fi',
                              'unit.linkings.taxon.vernacularName.sv'],
    'eventDate':             ['gathering.displayDateTime',
                              'gathering.eventDate.begin'],
    # 'class' is filled from the request-side filter (we know what we asked for)
}

OUTPUT_COLUMNS = list(FIELD_MAP.keys()) + ['class']


def _get_token(args: argparse.Namespace) -> str:
    tok = args.access_token or os.environ.get('FINBIF_TOKEN')
    if not tok:
        sys.exit(
            'No FinBIF token. Pass --access-token or set FINBIF_TOKEN.\n'
            'Register: '
            'curl -X POST -d email=<your@addr> https://api.laji.fi/v0/api-users'
        )
    return tok


_RETRY_STATUSES = (429, 500, 502, 503, 504)
_RETRY_EXCEPTIONS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)


def _http_get(url: str, params: Dict[str, str], headers: Dict[str, str],
              timeout: int = 300, max_retries: int = 6) -> requests.Response:
    """GET with exponential-backoff retries on transient errors.

    Retries on connection errors, timeouts, and 5xx / 429 responses.
    Raises ``RuntimeError`` after ``max_retries`` failures or on a
    permanent 4xx response.
    """
    last_err: Optional[str] = None
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
        except _RETRY_EXCEPTIONS as exc:
            last_err = f'{type(exc).__name__}: {exc}'
        else:
            if r.status_code == 200:
                return r
            if r.status_code in _RETRY_STATUSES:
                last_err = f'HTTP {r.status_code}: {r.text[:200]}'
            else:
                # Permanent error — don't retry.
                raise RuntimeError(f'FinBIF API {r.status_code}: {r.text[:300]}')
        sleep_s = min(60, 2 ** attempt) + 0.5 * attempt
        print(
            f'  [retry] attempt {attempt + 1}/{max_retries} after {last_err}; '
            f'sleeping {sleep_s:.1f}s', file=sys.stderr,
        )
        time.sleep(sleep_s)
    raise RuntimeError(f'failed after {max_retries} retries: {last_err}')


def _state_path(out_path: Path) -> Path:
    """Sidecar file path holding download progress."""
    return out_path.with_suffix(out_path.suffix + '.state.json')


def _load_state(out_path: Path, query_signature: str) -> Optional[Dict]:
    """Load resume state if it matches *query_signature*; else None."""
    p = _state_path(out_path)
    if not p.is_file():
        return None
    try:
        st = json.loads(p.read_text(encoding='utf-8'))
    except Exception as exc:
        print(f'  [resume] cannot parse {p.name}: {exc}; starting fresh',
              file=sys.stderr)
        return None
    if st.get('query_signature') != query_signature:
        print(f'  [resume] state file {p.name} is for a different query '
              f'(want {query_signature!r}, found {st.get("query_signature")!r}); '
              f'starting fresh', file=sys.stderr)
        return None
    return st


def _save_state(out_path: Path, query_signature: str, last_page: int,
                last_page_total: Optional[int], n_kept: int) -> None:
    p = _state_path(out_path)
    tmp = p.with_suffix(p.suffix + '.tmp')
    payload = {
        'query_signature': query_signature,
        'last_completed_page': last_page,
        'last_page_known_total': last_page_total,
        'records_written': n_kept,
        'updated': _dt.datetime.now(_dt.timezone.utc).isoformat(),
    }
    tmp.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    tmp.replace(p)


def _clear_state(out_path: Path) -> None:
    p = _state_path(out_path)
    try:
        p.unlink()
    except FileNotFoundError:
        pass


def _query_signature(params: Dict[str, str]) -> str:
    """A stable string identifying the *content* of a query for resume.

    Built from the params that actually affect the result set — pageSize
    and page are excluded since they only affect how the data is sliced.
    """
    sig_keys = ('time', 'informalTaxonGroupId', 'coordinateAccuracyMax',
                'qualityIssues')
    return '&'.join(f'{k}={params[k]}' for k in sig_keys if k in params)


def _walk(d, dotted: str):
    """Look up a dotted path in a nested dict."""
    cur = d
    for part in dotted.split('.'):
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, list) and part.isdigit():
            i = int(part)
            cur = cur[i] if i < len(cur) else None
        else:
            return None
    return cur


def _record_to_row(rec: Dict, default_class: str) -> Optional[Dict[str, str]]:
    out: Dict[str, str] = {}
    for col, candidates in FIELD_MAP.items():
        val = None
        for c in candidates:
            val = _walk(rec, c)
            if val not in (None, ''):
                break
        out[col] = '' if val is None else str(val).replace(',', ';')
    out['class'] = default_class
    if not (out['latitude'] and out['longitude']):
        return None
    return out


def _build_params(args: argparse.Namespace, *, time_override: Optional[str] = None) -> Dict[str, str]:
    params: Dict[str, str] = {
        'pageSize': str(args.page_size),
        'coordinateAccuracyMax': str(args.max_accuracy),
        'qualityIssues': 'NO_ISSUES',
    }
    if time_override is not None:
        params['time'] = time_override
    elif args.year_from is not None or args.year_to is not None:
        a = str(args.year_from) if args.year_from is not None else '1900'
        b = str(args.year_to)   if args.year_to   is not None else '2100'
        params['time'] = f'{a}/{b}'
    if args.classes:
        ids = []
        for c in args.classes:
            gid = CLASS_TO_GROUP.get(c.lower())
            if not gid:
                sys.exit(f'Unknown class {c!r}. Known: {sorted(CLASS_TO_GROUP)}')
            ids.append(gid)
        params['informalTaxonGroupId'] = ','.join(ids)
    return params


def _resolve_time_window(args: argparse.Namespace, out_path: Path) -> Optional[str]:
    """Determine the effective FinBIF ``time`` window.

    Returns the ``time`` parameter value (e.g. ``2009-12-25/2009`` after
    backing off ``--resume-overlap-days`` from the latest eventDate in
    *out_path*), or None to fall through to year-only defaults.

    If the existing file already covers the entire requested window
    (max_date >= year_to year-end), returns the sentinel string ``''``
    indicating "nothing to do".
    """
    last = _last_event_date(out_path) if out_path else None
    if not last:
        return None
    overlap = max(int(args.resume_overlap_days), 0)
    end_year = args.year_to if args.year_to is not None else 2100
    end_date = f'{end_year}-12-31'
    if last >= end_date:
        return ''  # already complete
    new_start = _shift_iso(last, -overlap)
    # Don't move start before the requested year_from.
    floor = f'{args.year_from}-01-01' if args.year_from is not None else '1900-01-01'
    if new_start < floor:
        new_start = floor
    print(
        f'  [resume] existing file {out_path.name} latest eventDate={last}; '
        f'continuing from {new_start} (overlap={overlap}d)',
        file=sys.stderr,
    )
    return f'{new_start}/{end_year}'


def _paginated(url: str, params: Dict[str, str], headers: Dict[str, str],
               start_page: int = 1) -> Iterable[Tuple[int, int, List[Dict]]]:
    """Yield ``(page_number, last_page, results)`` from each successful page.

    Caller is responsible for persisting state after consuming each page.
    """
    page = start_page
    while True:
        params['page'] = str(page)
        r = _http_get(url, params, headers, timeout=300, max_retries=6)
        data = r.json()
        results = data.get('results', [])
        last_page = data.get('lastPage') or page
        if not results:
            return
        yield page, last_page, results
        if page >= last_page:
            return
        page += 1


def _stream_via_list(args: argparse.Namespace, params: Dict[str, str],
                     headers: Dict[str, str]) -> int:
    """Page through ``/warehouse/query/list`` and write records to CSV.

    Resumes via the per-output ``.state.json`` sidecar: if a state file
    exists for the same query, picks up at ``last_completed_page + 1``
    in append mode; otherwise starts fresh and writes a header.
    """
    url = f'{API_BASE}/warehouse/query/list'
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    use_gz = str(out_path).endswith('.gz')

    sig = _query_signature(params)
    state = _load_state(out_path, sig)
    start_page = (state['last_completed_page'] + 1) if state else 1
    n_kept = state['records_written'] if state else 0
    append = state is not None
    if state:
        print(f'  [resume] {out_path.name}: continuing at page {start_page} '
              f'(records already written: {n_kept:,})', file=sys.stderr)

    mode = 'at' if append else 'wt'
    f = gzip.open(out_path, mode, encoding='utf-8') if use_gz else \
        out_path.open('a' if append else 'w', encoding='utf-8')
    default_class = (args.classes[0] if args.classes else '').lower()
    try:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        if not append:
            writer.writeheader()
        n_total = 0
        last_page_known: Optional[int] = None
        for page, last_page, results in _paginated(url, params, headers,
                                                    start_page=start_page):
            last_page_known = last_page
            for rec in results:
                n_total += 1
                row = _record_to_row(rec, default_class)
                if row is None:
                    continue
                writer.writerow(row)
                n_kept += 1
            f.flush()
            _save_state(out_path, sig, page, last_page_known, n_kept)
            if page % 10 == 0 or page == last_page:
                print(f'  page={page}/{last_page}  fetched={n_total:,}  '
                      f'total_kept={n_kept:,}', file=sys.stderr)
        # Successful completion → drop the state sidecar so the next
        # invocation knows the file is done (vs partial mid-run).
        _clear_state(out_path)
        return n_kept
    finally:
        f.close()


def _async_download(args: argparse.Namespace, params: Dict[str, str],
                    headers: Dict[str, str]) -> Path:
    """Submit a /warehouse/download job, poll, save the resulting archive.

    Suitable for multi-million-record pulls where /list pagination would
    take too long.
    """
    submit = requests.post(f'{API_BASE}/warehouse/download',
                           params=params, headers=headers, timeout=120)
    if submit.status_code not in (200, 201, 202):
        raise RuntimeError(f'submit failed {submit.status_code}: {submit.text[:400]}')
    info = submit.json()
    request_id = info.get('id') or info.get('requestId') or info.get('downloadRequestId')
    if not request_id:
        raise RuntimeError(f'no request id in response: {info}')
    print(f'  submitted FinBIF download request id={request_id} — polling...',
          file=sys.stderr)
    poll_url = f'{API_BASE}/warehouse/download/{request_id}'
    while True:
        r = requests.get(poll_url, headers=headers, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f'poll failed {r.status_code}: {r.text[:300]}')
        info = r.json()
        status = info.get('status', '?')
        if status in ('COMPLETED', 'READY', 'OK'):
            break
        if status in ('FAILED', 'ERROR'):
            raise RuntimeError(f'FinBIF download failed: {info}')
        print(f'    status={status} — waiting {args.poll_interval}s ...',
              file=sys.stderr)
        time.sleep(args.poll_interval)

    file_url = info.get('downloadUrl') or info.get('uri')
    if not file_url:
        raise RuntimeError(f'no download URL in {info}')
    out_zip = Path(args.output).with_suffix('.zip')
    print(f'  downloading from {file_url} → {out_zip}', file=sys.stderr)
    with requests.get(file_url, headers=headers, stream=True, timeout=600) as r:
        r.raise_for_status()
        with out_zip.open('wb') as f:
            for chunk in r.iter_content(chunk_size=2 ** 20):
                f.write(chunk)
    return out_zip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--output', '-o', required=True,
                        help='Output CSV path (use .gz to gzip).')
    parser.add_argument('--classes', nargs='+', default=['Aves'],
                        help=f'Taxonomic classes. '
                             f'Recognized: {sorted(CLASS_TO_GROUP)}.')
    parser.add_argument('--year-from', type=int, default=2010)
    parser.add_argument('--year-to', type=int, default=None,
                        help='End year inclusive (default: open-ended).')
    parser.add_argument('--max-accuracy', type=int, default=5000,
                        help='Drop records with coordinate accuracy worse than '
                             'this many metres (default 5000 = 5 km).')
    parser.add_argument('--mode', choices=['list', 'async'], default='list',
                        help="'list' streams from /warehouse/query/list. "
                             "'async' submits a /warehouse/download job.")
    parser.add_argument('--page-size', type=int, default=1000)
    parser.add_argument('--poll-interval', type=int, default=15)
    parser.add_argument('--access-token',
                        help='FinBIF access token. Or set FINBIF_TOKEN env var.')
    parser.add_argument('--restart', action='store_true',
                        help='Ignore any existing .state.json and re-download '
                             'from scratch (overwrites the output).')
    args = parser.parse_args()

    args._token = _get_token(args)
    headers = {'Authorization': f'Bearer {args._token}'}

    out_path = Path(args.output)
    if args.restart:
        _clear_state(out_path)
        if out_path.exists():
            out_path.unlink()

    params = _build_params(args)

    print(f'classes={args.classes}  time={params.get("time","-")}  '
          f'max_accuracy={args.max_accuracy}m  mode={args.mode}',
          file=sys.stderr)

    if args.mode == 'async':
        path = _async_download(args, params, headers)
        print(f'\nFinBIF archive saved to {path}\n'
              f'  → unzip and pass through utils/gbifutils.py to merge with the rest.')
    else:
        n = _stream_via_list(args, params, headers)
        print(f'\nWrote {n:,} records to {args.output}')


if __name__ == '__main__':
    main()
