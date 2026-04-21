"""Request and download a GBIF Darwin Core Archive via the Occurrence Download API.

Builds a predicate (country / bounding box / polygon + optional
state-province + year range + coordinate/geo-issue filters), submits a
download request, polls until the archive is ready, then streams the zip
to disk. Output is a DwC-A zip consumable by ``utils/gbifutils.py``.

Requires a registered GBIF account (https://www.gbif.org/user/profile).
Credentials are read from environment variables:
    GBIF_USERNAME (or GBIF_USER), GBIF_PASSWORD, GBIF_EMAIL

A .env file in the repo root is honored via python-dotenv.

Usage:
    # Norway, last 10 years (defaults)
    python utils/gbif_download.py --output data/gbif_norway_10y.zip

    # Sweden, Aves only, 2020-2025
    python utils/gbif_download.py --country SE \\
        --start-year 2020 --end-year 2025 \\
        --taxon-keys aves \\
        --output data/gbif_sweden_birds.zip

    # Multiple countries: Germany + Poland, birds, last 10 years
    python utils/gbif_download.py --country DE,PL --taxon-keys aves \\
        --output data/gbif_de-pl_birds.zip

    # Northern Germany + Poland via bounding box (lat 52-55.5)
    python utils/gbif_download.py --country DE,PL \\
        --bbox 6,52,24,55.5 --taxon-keys aves \\
        --output data/gbif_north-de-pl_birds.zip

    # Arbitrary polygon (WKT); drop --country to search worldwide within it
    python utils/gbif_download.py --country "" \\
        --polygon "POLYGON((6 52, 24 52, 24 55.5, 6 55.5, 6 52))" \\
        --taxon-keys aves --output data/gbif_custom_aves.zip
"""

import argparse
import datetime as dt
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import requests
from tqdm import tqdm

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

LOG = logging.getLogger(__name__)

GBIF_API = "https://api.gbif.org/v1"

# Common GBIF backbone taxonKeys (for --taxon-keys convenience).
# GBIF's TAXON_KEY predicate matches all descendants of the given key, so a
# kingdom-level key like Plantae pulls in every plant species underneath it.
BACKBONE_KEYS = {
    # Animal classes
    'aves':      212,
    'amphibia':  131,
    'insecta':   216,
    'mammalia':  359,
    'reptilia':  358,
    # Kingdoms
    'plantae':   6,
    'fungi':     5,
}


def bbox_to_wkt(west: float, south: float, east: float, north: float) -> str:
    """Convert a lon/lat bounding box to a WKT POLYGON.

    GBIF's ``GEOMETRY`` predicate expects WKT with coordinates in lon/lat
    order and a closed ring (first == last). Rings must be counter-clockwise
    for the outer boundary — GBIF is strict about this; a CW ring silently
    matches the *complement* of the intended area.
    """
    if east <= west:
        raise ValueError(f"bbox: east ({east}) must be > west ({west})")
    if north <= south:
        raise ValueError(f"bbox: north ({north}) must be > south ({south})")
    # Counter-clockwise: SW → SE → NE → NW → SW
    return (
        f"POLYGON(({west} {south}, {east} {south}, "
        f"{east} {north}, {west} {north}, {west} {south}))"
    )


def build_predicate(
    countries: Optional[List[str]],
    start_year: int,
    end_year: int,
    taxon_keys: Optional[List[int]] = None,
    geometry_wkt: Optional[str] = None,
    state_province: Optional[str] = None,
) -> dict:
    """Build a GBIF download predicate for the given filters.

    ``countries`` accepts 0, 1, or many ISO alpha-2 codes. Pass an empty
    list (or ``None``) together with ``geometry_wkt`` to search worldwide
    within the polygon. Multiple countries are ANDed with the polygon, so
    a country + bbox combination clips the polygon to just those countries'
    records.
    """
    preds = [
        {"type": "greaterThanOrEquals", "key": "YEAR", "value": str(start_year)},
        {"type": "lessThanOrEquals", "key": "YEAR", "value": str(end_year)},
        {"type": "equals", "key": "HAS_COORDINATE", "value": "true"},
        {"type": "equals", "key": "HAS_GEOSPATIAL_ISSUE", "value": "false"},
        {"type": "equals", "key": "OCCURRENCE_STATUS", "value": "PRESENT"},
    ]

    # Country filter — single value uses equals, multiple use "in".
    if countries:
        normalised = [c.strip().upper() for c in countries if c.strip()]
        if len(normalised) == 1:
            preds.append({"type": "equals", "key": "COUNTRY",
                          "value": normalised[0]})
        elif len(normalised) > 1:
            preds.append({"type": "in", "key": "COUNTRY",
                          "values": normalised})

    # Geometry (bbox or polygon, already normalised to WKT upstream).
    # GBIF's "within" predicate uses a top-level "geometry" field rather
    # than "key"+"value". The API rejects the polygon if its outer ring is
    # clockwise — `bbox_to_wkt` above emits CCW.
    if geometry_wkt:
        preds.append({"type": "within", "geometry": geometry_wkt})

    # State/province — unreliable at the publisher level (spellings drift),
    # but sometimes useful as a post-filter. Case-sensitive exact match.
    if state_province:
        preds.append({"type": "equals", "key": "STATE_PROVINCE",
                      "value": state_province})

    if taxon_keys:
        if len(taxon_keys) == 1:
            preds.append({"type": "equals", "key": "TAXON_KEY",
                          "value": str(taxon_keys[0])})
        else:
            preds.append({
                "type": "in", "key": "TAXON_KEY",
                "values": [str(k) for k in taxon_keys],
            })
    return {"type": "and", "predicates": preds}


def request_download(
    predicate: dict, user: str, password: str, email: str, fmt: str = "DWCA",
) -> str:
    """Submit a download request and return the download key."""
    body = {
        "creator": user,
        "notificationAddresses": [email],
        "sendNotification": False,
        "format": fmt,
        "predicate": predicate,
    }
    r = requests.post(
        f"{GBIF_API}/occurrence/download/request",
        json=body, auth=(user, password), timeout=60,
    )
    if r.status_code not in (200, 201):
        raise RuntimeError(
            f"Download request failed ({r.status_code}): {r.text}"
        )
    return r.text.strip()


def poll_until_ready(key: str, poll_interval: int = 30) -> dict:
    """Poll the download metadata until status is terminal. Return final meta."""
    terminal = {"SUCCEEDED", "CANCELLED", "KILLED", "FAILED", "FILE_ERASED"}
    url = f"{GBIF_API}/occurrence/download/{key}"
    last_status = None
    while True:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        meta = r.json()
        status = meta.get("status", "UNKNOWN")
        if status != last_status:
            records = meta.get("totalRecords")
            suffix = f" ({records:,} records)" if records else ""
            LOG.info("Download %s: %s%s", key, status, suffix)
            last_status = status
        if status in terminal:
            return meta
        time.sleep(poll_interval)


def resolve_output_path(
    out: Path,
    countries: Optional[List[str]],
    start_year: int,
    end_year: int,
    has_geometry: bool = False,
) -> Path:
    """Normalise ``--output``: if it's a directory (or has no extension),
    append a default filename. Always create parent directories.

    Avoids the failure mode of writing to a bare directory name (e.g.
    ``data/``) which would clobber an existing directory or symlink.
    """
    # Follows symlinks: True for symlink→dir.
    looks_like_dir = out.is_dir() or out.suffix == ""
    if looks_like_dir:
        out.mkdir(parents=True, exist_ok=True)
        if countries:
            slug = "-".join(c.lower() for c in countries)
        else:
            slug = "custom"
        if has_geometry:
            slug += "-bbox"
        default_name = f"gbif_{slug}_{start_year}-{end_year}.zip"
        out = out / default_name
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
    return out


def stream_to_file(url: str, out_path: Path, auth=None) -> None:
    """Stream a URL to disk with a progress bar, atomic rename on completion."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    with requests.get(url, stream=True, auth=auth, timeout=300) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length") or 0)
        with open(tmp, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True,
            desc=f"Downloading {out_path.name}",
        ) as pbar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    tmp.rename(out_path)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    this_year = dt.date.today().year
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--country", default="NO",
                        help="ISO 3166-1 alpha-2 country code(s). Comma-separated "
                             "for multiple (e.g. 'DE,PL'). Pass an empty string "
                             "('--country \"\"') to skip the country filter when "
                             "using --bbox or --polygon. Default: NO")
    parser.add_argument("--bbox",
                        help="Lon/lat bounding box as W,S,E,N (degrees). "
                             "Example: '6,52,24,55.5' for Northern Germany + "
                             "Poland. Converted to a CCW WKT POLYGON. "
                             "Mutually exclusive with --polygon.")
    parser.add_argument("--polygon",
                        help="WKT POLYGON / MULTIPOLYGON (lon/lat, CCW outer ring). "
                             "Mutually exclusive with --bbox. Pass a file path "
                             "prefixed with '@' to read WKT from a file, e.g. "
                             "'@data/kreis_rostock.wkt'.")
    parser.add_argument("--state-province",
                        help="Optional STATE_PROVINCE filter (case-sensitive exact "
                             "match). Noisy at the publisher level — prefer --bbox "
                             "or --polygon for reproducibility.")
    parser.add_argument("--start-year", type=int, default=this_year - 10,
                        help=f"Inclusive start year (default: {this_year - 10})")
    parser.add_argument("--end-year", type=int, default=this_year,
                        help=f"Inclusive end year (default: {this_year})")
    parser.add_argument("--taxon-keys", nargs="*", default=None,
                        help="GBIF backbone taxonKeys to filter. Accepts integers "
                             "or names: "
                             + ", ".join(f"{k}={v}" for k, v in BACKBONE_KEYS.items()))
    parser.add_argument("--format", default="DWCA", choices=["DWCA", "SIMPLE_CSV"],
                        help="Archive format (default: DWCA, matches gbifutils.py)")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output zip path, or a directory into which a "
                             "default filename (gbif_<country>_<start>-<end>.zip) "
                             "is written. Parent directories are created.")
    parser.add_argument("--poll-interval", type=int, default=30,
                        help="Seconds between status polls (default: 30)")
    parser.add_argument("--user",
                        default=os.environ.get("GBIF_USERNAME")
                                or os.environ.get("GBIF_USER"))
    parser.add_argument("--password", default=os.environ.get("GBIF_PASSWORD"))
    parser.add_argument("--email", default=os.environ.get("GBIF_EMAIL"))
    parser.add_argument("--existing-key",
                        help="Skip request + polling, download an already-"
                             "prepared download by key")
    args = parser.parse_args()

    if not (args.user and args.password and args.email):
        parser.error("GBIF credentials missing. Set GBIF_USERNAME / "
                     "GBIF_PASSWORD / GBIF_EMAIL env vars (or pass "
                     "--user/--password/--email). "
                     "Register at https://www.gbif.org/user/profile")

    # Countries: comma-separated, empty string opts out entirely.
    countries: List[str] = []
    if args.country is not None and args.country != "":
        countries = [c.strip().upper() for c in args.country.split(",") if c.strip()]

    # Geometry: bbox XOR polygon. Both feed into the same WKT predicate.
    if args.bbox and args.polygon:
        parser.error("--bbox and --polygon are mutually exclusive")
    geometry_wkt: Optional[str] = None
    if args.bbox:
        try:
            parts = [float(x) for x in args.bbox.split(",")]
        except ValueError:
            parser.error(f"--bbox must be four comma-separated numbers, got {args.bbox!r}")
        if len(parts) != 4:
            parser.error(f"--bbox expects exactly 4 values (W,S,E,N); got {len(parts)}")
        geometry_wkt = bbox_to_wkt(*parts)
    elif args.polygon:
        raw = args.polygon
        if raw.startswith("@"):
            path = Path(raw[1:])
            if not path.is_file():
                parser.error(f"--polygon @file not found: {path}")
            raw = path.read_text().strip()
        geometry_wkt = raw

    if not countries and not geometry_wkt:
        parser.error("No geography filter — pass --country, --bbox, or --polygon "
                     "(passing '--country \"\"' without geometry would download "
                     "the entire world, which is almost certainly a mistake).")

    # Resolve output: create intermediate dirs, and if the user passed a
    # directory (or a bare name with no extension) append a default filename
    # rather than clobbering the directory / symlink.
    args.output = resolve_output_path(
        args.output, countries, args.start_year, args.end_year,
        has_geometry=geometry_wkt is not None,
    )
    LOG.info("Output: %s", args.output)

    # Resolve taxon keys (accept names or ints)
    taxon_keys: Optional[List[int]] = None
    if args.taxon_keys:
        taxon_keys = []
        for tk in args.taxon_keys:
            if tk.isdigit():
                taxon_keys.append(int(tk))
            else:
                key = BACKBONE_KEYS.get(tk.lower())
                if key is None:
                    parser.error(f"Unknown taxon name: {tk}. "
                                 f"Known: {sorted(BACKBONE_KEYS)}")
                taxon_keys.append(key)

    if args.existing_key:
        key = args.existing_key
        LOG.info("Using existing download key: %s", key)
    else:
        predicate = build_predicate(
            countries=countries,
            start_year=args.start_year,
            end_year=args.end_year,
            taxon_keys=taxon_keys,
            geometry_wkt=geometry_wkt,
            state_province=args.state_province,
        )
        LOG.info("Submitting download: countries=%s bbox/poly=%s state=%s "
                 "years=%d-%d taxon_keys=%s",
                 countries or "(worldwide)",
                 "yes" if geometry_wkt else "no",
                 args.state_province or "-",
                 args.start_year, args.end_year, taxon_keys)
        key = request_download(predicate, args.user, args.password,
                               args.email, fmt=args.format)
        LOG.info("Download key: %s", key)
        LOG.info("Status page: https://www.gbif.org/occurrence/download/%s", key)

    meta = poll_until_ready(key, poll_interval=args.poll_interval)
    if meta.get("status") != "SUCCEEDED":
        LOG.error("Download did not succeed: %s", meta.get("status"))
        return 1

    records = meta.get("totalRecords", 0)
    doi = meta.get("doi")
    size_mb = (meta.get("size") or 0) / (1024 * 1024)
    LOG.info("Ready: %s records, %.1f MB. DOI: %s", f"{records:,}", size_mb, doi)

    download_url = f"{GBIF_API}/occurrence/download/request/{key}.zip"
    stream_to_file(download_url, args.output)
    LOG.info("Wrote %s", args.output)
    LOG.info("Cite as: GBIF.org (%s) GBIF Occurrence Download %s",
             dt.date.today().isoformat(), doi)
    return 0


if __name__ == "__main__":
    sys.exit(main())
