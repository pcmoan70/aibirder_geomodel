#!/usr/bin/env python3
"""
download_sweden_dtm.py — pull Swedish DTM tiles for further processing
alongside the Norwegian DTM10 set.

Sweden's open terrain-model landscape in 2025:

  * **Lantmäteriet "Höjddata, grid 1"** (NH 1 m) — fine 1 m DTM, CC0,
    rolled out nationally via the open-data portal at
    https://dataforsorjning.lantmateriet.se/. Downloadable as GeoTIFF
    tiles over the STAC API + download endpoints at `api.lantmateriet.se`.
    Requires a free API consumer token (register at Geotorget — no
    payment) that the script reads from `--api-token` or the
    `LANTMATERIET_TOKEN` env var.

  * **Höjddata, grid 50+** — open 50 m DTM, old product, quick smoke
    test. Downloadable without a token from the HTTPS public mirror.

  * **Copernicus GLO-30** — 30 m global, open, no auth. Good fallback
    when you don't have a Lantmäteriet token or only need coarse
    elevation.

Defaults to the Lantmäteriet 1 m product (resample locally to 10 m if
you want to pair-match Norwegian DTM10). Switch with `--source`.

Output: a flat folder of per-tile `.tif` / `.zip` files, naming kept
from the source so re-runs are idempotent (already-downloaded tiles are
skipped).

Usage:
    export LANTMATERIET_TOKEN=...
    python3 download_sweden_dtm.py --out /media/pc/ext4TB/DTM_SE

    # 30 m Copernicus fallback (no token):
    python3 download_sweden_dtm.py --out /media/pc/ext4TB/DTM_SE_cop --source copernicus

Dependencies:  requests tqdm
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Lantmäteriet open API — "Höjddata, grid 1" (NH 1 m, CC0).
# STAC browse + atom-download pattern documented at
# https://dataforsorjning.lantmateriet.se/dokumentation.
# URLs verified Jan 2025; if these change the user can point `--stac-url`
# + `--download-base` at whatever the portal docs now say.
# ---------------------------------------------------------------------------
LM_STAC_URL = "https://api.lantmateriet.se/stac-hojd-grid1/v1/collections/hojd-grid1/items"
LM_PAGE_LIMIT = 100
# Sweden's national bounding box in WGS84 (lon/lat, EPSG:4326).
SWEDEN_BBOX = (10.9, 55.3, 24.2, 69.1)

# ---------------------------------------------------------------------------
# Copernicus GLO-30 fallback (AWS Open Data, no auth).
# Tiles are 1°×1° named by their SW corner.
# ---------------------------------------------------------------------------
COP_BASE = (
    "https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/"
    "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM/"
    "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM.tif"
)


# ---------------------------------------------------------------------------
# Lantmäteriet path
# ---------------------------------------------------------------------------
def lantmateriet_tile_urls(
    token: str,
    bbox: tuple[float, float, float, float],
    stac_url: str = LM_STAC_URL,
) -> Iterator[tuple[str, str]]:
    """Yield (filename, download_url) pairs for every tile the STAC
    item list returns inside `bbox`. Paginates with `next` links."""
    sess = requests.Session()
    sess.headers.update({
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    })
    params = {
        "bbox": ",".join(str(v) for v in bbox),
        "limit": LM_PAGE_LIMIT,
    }
    url = stac_url
    while url:
        resp = sess.get(url, params=params, timeout=60)
        if resp.status_code == 401:
            raise RuntimeError(
                "Lantmäteriet API returned 401 Unauthorized — check your "
                "consumer token. Register (free) at https://geotorget.lantmateriet.se/."
            )
        resp.raise_for_status()
        payload = resp.json()
        for item in payload.get("features", []):
            # Each feature has an `assets` dict with the GeoTIFF URL.
            for asset_key, asset in (item.get("assets") or {}).items():
                href = asset.get("href")
                if not href:
                    continue
                # Prefer explicit GeoTIFF assets.
                media = (asset.get("type") or "").lower()
                if "tiff" not in media and not href.endswith(".tif"):
                    continue
                name = Path(href).name or f"{item['id']}.tif"
                yield name, href
        # STAC pagination via `links[rel=next]`.
        next_link = next((l for l in payload.get("links", []) if l.get("rel") == "next"), None)
        if next_link:
            url = next_link["href"]
            params = None  # next link already contains the cursor
        else:
            url = None
        # Reset query params after first page since next link is absolute.


# ---------------------------------------------------------------------------
# Copernicus fallback
# ---------------------------------------------------------------------------
def copernicus_tile_urls(bbox: tuple[float, float, float, float]) -> Iterator[tuple[str, str]]:
    min_lon, min_lat, max_lon, max_lat = bbox
    for lat in range(int(min_lat), int(max_lat) + 1):
        for lon in range(int(min_lon), int(max_lon) + 1):
            ns = "N" if lat >= 0 else "S"
            ew = "E" if lon >= 0 else "W"
            url = COP_BASE.format(ns=ns, lat=abs(lat), ew=ew, lon=abs(lon))
            name = Path(url).name
            yield name, url


# ---------------------------------------------------------------------------
# Parallel download helper
# ---------------------------------------------------------------------------
def download_one(url: str, dest: Path, session: requests.Session, token: str | None) -> tuple[Path, bool, str]:
    """Stream-download `url` to `dest`. Returns (path, ok, msg).
    Skips when `dest` already exists with non-zero size (idempotent)."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest, True, "skipped-exists"
    tmp = dest.with_suffix(dest.suffix + ".part")
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        with session.get(url, headers=headers, stream=True, timeout=300) as r:
            if r.status_code == 404:
                return dest, False, "404 (tile missing)"
            r.raise_for_status()
            tmp.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
        tmp.rename(dest)
        return dest, True, "ok"
    except requests.exceptions.RequestException as e:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        return dest, False, f"error: {e}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="/media/pc/ext4TB/DTM_SE",
                   help="Output folder for tiles.")
    p.add_argument("--source", choices=("lantmateriet", "copernicus"),
                   default="lantmateriet",
                   help="Which open-data provider to pull from. Lantmäteriet gives "
                        "1 m tiles (token required); Copernicus gives 30 m globally.")
    p.add_argument("--api-token", default=None,
                   help="Lantmäteriet API consumer token. Falls back to $LANTMATERIET_TOKEN.")
    p.add_argument("--stac-url", default=LM_STAC_URL,
                   help="Override the Lantmäteriet STAC endpoint if the URL moves.")
    p.add_argument("--bbox", default=None,
                   help='Override bbox "min_lon,min_lat,max_lon,max_lat" (default: Sweden).')
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel downloads (be kind to the servers).")
    p.add_argument("--dry-run", action="store_true",
                   help="Only list tile URLs; download nothing.")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger(__name__)

    if args.bbox:
        bbox = tuple(float(v) for v in args.bbox.split(","))
        if len(bbox) != 4:
            raise SystemExit("--bbox must be 4 comma-separated floats")
    else:
        bbox = SWEDEN_BBOX

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    token = args.api_token or os.environ.get("LANTMATERIET_TOKEN")

    if args.source == "lantmateriet":
        if not token:
            log.error(
                "Lantmäteriet open-data download needs a consumer token. "
                "Register (free) at https://geotorget.lantmateriet.se/ and "
                "pass --api-token or set $LANTMATERIET_TOKEN. Run with "
                "--source copernicus to skip this and pull 30 m global tiles."
            )
            sys.exit(2)
        url_iter = lantmateriet_tile_urls(token, bbox, stac_url=args.stac_url)
    else:
        url_iter = copernicus_tile_urls(bbox)

    jobs: list[tuple[str, str]] = []
    log.info("Enumerating tiles from %s…", args.source)
    try:
        for name, url in url_iter:
            jobs.append((name, url))
    except Exception as e:  # noqa: BLE001
        log.error("Failed to enumerate tiles: %s", e)
        sys.exit(1)

    log.info("Tiles to fetch: %d", len(jobs))
    if args.dry_run:
        for name, url in jobs:
            print(f"{name}\t{url}")
        return

    session = requests.Session()
    ok = skipped = missing = failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(download_one, url, out_dir / name, session, token if args.source == "lantmateriet" else None): name
            for name, url in jobs
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Download"):
            dest, success, msg = fut.result()
            if success and msg == "ok":
                ok += 1
            elif success and msg.startswith("skipped"):
                skipped += 1
            elif msg.startswith("404"):
                missing += 1
            else:
                failed += 1
                log.warning("FAIL %s — %s", dest.name, msg)

    log.info("Downloaded %d · skipped %d · missing-on-server %d · failed %d → %s",
             ok, skipped, missing, failed, out_dir)


if __name__ == "__main__":
    main()
