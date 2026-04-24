#!/usr/bin/env python3
"""
altitude_from_dtm.py — populate an `altitude` column in
combined.parquet from the Norwegian DTM10 tile set.

The DTM10 archive (Kartverket) is a folder of ZIPs. Each ZIP contains
one `.dem` tile in EPSG:25833 (ETRS89 / UTM 33N). GDAL reads them via
the `/vsizip/...` virtual filesystem — no manual extraction needed.

The parquet has `h3_index` rows (H3 cell ids, no raw lat/lon). We take
each cell's centroid, project to UTM 33N, and sample the first DTM
tile whose bounding box contains the point. First-hit wins; a cell
that falls outside every tile ends with `altitude = NaN`.

Usage:
    python3 altitude_from_dtm.py                               # defaults
    python3 altitude_from_dtm.py --overwrite-existing          # reset + redo
    python3 altitude_from_dtm.py --out /tmp/out.parquet        # custom output

Dependencies:  pandas pyarrow rasterio pyproj h3 tqdm numpy
"""
from __future__ import annotations

import argparse
import logging
import zipfile
from pathlib import Path

import h3
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.transform import rowcol
from tqdm import tqdm


def locate_raster_entry(zip_path: Path) -> str | None:
    """Return the first raster filename inside the zip, or None.

    Kartverket ships `.dem` (USGS DEM) inside the DTM10 zips, but the
    same script handles `.tif` / `.tiff` so it doesn't break if the
    archive layout ever changes."""
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                lower = name.lower()
                if lower.endswith((".dem", ".tif", ".tiff")):
                    return name
    except zipfile.BadZipFile:
        return None
    return None


def h3_centroids(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised H3 centroid → (lat, lon) in degrees.

    The `h3` Python library's API name changed between v3 (`h3_to_geo`)
    and v4 (`cell_to_latlng`). Probe once and dispatch."""
    if hasattr(h3, "cell_to_latlng"):          # v4
        fn = h3.cell_to_latlng
    elif hasattr(h3, "h3_to_geo"):             # v3
        fn = h3.h3_to_geo
    else:
        raise RuntimeError("Unsupported `h3` package — need v3 or v4.")
    lat = np.empty(len(indices), dtype=np.float64)
    lon = np.empty(len(indices), dtype=np.float64)
    for i, idx in enumerate(indices):
        la, lo = fn(idx)
        lat[i] = la
        lon[i] = lo
    return lat, lon


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dtm-dir", default="/media/pc/ext4TB/DTM10",
                        help="Folder with Kartverket DTM10 ZIP tiles.")
    parser.add_argument("--parquet", default="/media/pc/HD1/aibirder_model_data/combined.parquet",
                        help="Input observations parquet.")
    parser.add_argument("--out", default=None,
                        help="Output parquet (default: <input>_with_altitude.parquet).")
    parser.add_argument("--h3-col", default="h3_index")
    parser.add_argument("--overwrite-existing", action="store_true",
                        help="Rewrite altitude for every row, even if it already has one.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger(__name__)

    dtm_dir = Path(args.dtm_dir)
    parquet_path = Path(args.parquet)
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = parquet_path.with_name(parquet_path.stem + "_with_altitude" + parquet_path.suffix)

    log.info("Loading %s", parquet_path)
    df = pd.read_parquet(parquet_path)
    if args.h3_col not in df.columns:
        raise SystemExit(f"Parquet missing column {args.h3_col!r}")
    log.info("%s: %d rows", parquet_path.name, len(df))

    if "altitude" not in df.columns:
        df["altitude"] = np.nan
    altitudes = df["altitude"].to_numpy(dtype=np.float64, copy=True)

    log.info("Computing H3 centroids for %d rows", len(df))
    lat, lon = h3_centroids(df[args.h3_col].to_numpy())

    # Rows still needing a fix. The parquet already has `elevation_m`
    # from some coarser source; leave it alone — the new `altitude`
    # column is the DTM10-sourced one.
    if args.overwrite_existing:
        pending = np.ones(len(df), dtype=bool)
    else:
        pending = ~np.isfinite(altitudes)
    log.info("%d/%d rows need altitude", pending.sum(), len(df))

    # Norwegian DTM10 is entirely in EPSG:25833 (ETRS89 / UTM 33N).
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:25833", always_xy=True)
    x_utm = np.full(len(df), np.nan, dtype=np.float64)
    y_utm = np.full(len(df), np.nan, dtype=np.float64)
    if pending.any():
        idx = np.where(pending)[0]
        xs, ys = transformer.transform(lon[idx], lat[idx])
        x_utm[idx] = xs
        y_utm[idx] = ys

    tile_zips = sorted(dtm_dir.glob("*.zip"))
    if not tile_zips:
        raise SystemExit(f"No *.zip tiles found under {dtm_dir}")
    log.info("Scanning %d DTM tile archives", len(tile_zips))

    tiles_used = 0
    for zip_path in tqdm(tile_zips, desc="Tiles"):
        if not pending.any():
            break
        entry = locate_raster_entry(zip_path)
        if entry is None:
            log.debug("No raster in %s, skipping", zip_path.name)
            continue
        vsi = f"/vsizip/{zip_path}/{entry}"
        try:
            with rasterio.open(vsi) as src:
                left, bottom, right, top = src.bounds

                # Fast bbox filter over pending indices.
                idx = np.where(pending)[0]
                xs = x_utm[idx]
                ys = y_utm[idx]
                within = (xs >= left) & (xs <= right) & (ys >= bottom) & (ys <= top)
                if not within.any():
                    continue

                sel_idx = idx[within]
                sel_x = xs[within]
                sel_y = ys[within]

                rows, cols = rowcol(src.transform, sel_x, sel_y)
                rows = np.asarray(rows, dtype=np.int64)
                cols = np.asarray(cols, dtype=np.int64)

                h, w = src.height, src.width
                in_bounds = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
                if not in_bounds.any():
                    continue
                sel_idx = sel_idx[in_bounds]
                rows = rows[in_bounds]
                cols = cols[in_bounds]

                # Read the band masked so NoData comes through as np.nan.
                data = src.read(1, masked=True)
                samples = data[rows, cols]
                vals = np.ma.filled(samples, fill_value=np.nan).astype(np.float64)

                # Belt-and-braces: also reject the common DEM NoData
                # sentinels in case a tile doesn't declare one.
                bad = np.isclose(vals, -32768) | np.isclose(vals, -9999) | (vals < -1e5)
                vals = np.where(bad, np.nan, vals)

                write_mask = np.isfinite(vals)
                if write_mask.any():
                    altitudes[sel_idx[write_mask]] = vals[write_mask]
                    pending[sel_idx[write_mask]] = False
                    tiles_used += 1
                    log.debug("%s: filled %d altitudes", zip_path.name, write_mask.sum())
        except (rasterio.errors.RasterioIOError, rasterio.errors.RasterioError) as e:
            log.warning("Rasterio failed on %s: %s", zip_path.name, e)
        except Exception as e:  # noqa: BLE001
            log.warning("Failed on %s: %s", zip_path.name, e)

    filled = np.isfinite(altitudes).sum()
    missing = len(df) - filled
    log.info("Filled %d altitudes from %d tiles; %d still missing (outside DTM10 coverage).",
             filled, tiles_used, missing)

    df["altitude"] = altitudes
    log.info("Writing %s", out_path)
    df.to_parquet(out_path, index=False)
    log.info("Done.")


if __name__ == "__main__":
    main()
