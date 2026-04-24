#!/usr/bin/env python3
"""
geoid.py — populate a `geoid_m` column in combined.parquet from the
EGM2008 1' geoid grid.

Why:
    GPS measures ellipsoidal height `h` (WGS84). Orthometric elevation
    (height above mean sea level) is `H = h - N`, where N is the
    geoid-ellipsoid separation for that lat/lon. The 1′ EGM2008 grid
    is the standard high-resolution source for N over land.

Source:
    NGA's EGM2008 1' geoid grid, distributed as a PGM inside
    `/media/pc/ext4TB/geoids_EGM2008 - 1'.zip`. Values are 16-bit
    unsigned with the standard NGA encoding:
        N_meters = raw * 0.003 - 108.0

Grid:
    21600 cols × 10801 rows, 1′ resolution. Row 0 = +90°N, row last
    = -90°S, column 0 = 0°E, column last = 360°E (wraps). We decode
    the actual dimensions from the PGM header so it still works if
    NGA ever redistributes at a different resolution.

Input:
    `combined.parquet` uses H3 cell ids (`h3_index`) rather than raw
    lat/lon, so we compute each cell's centroid before sampling.

Output:
    A new parquet with the same rows plus a `geoid_m` column
    (float32). Saved alongside the input by default.

Dependencies:
    pandas pyarrow numpy h3 tqdm
"""
from __future__ import annotations

import argparse
import logging
import mmap
import zipfile
from pathlib import Path

import h3
import numpy as np
import pandas as pd
from tqdm import tqdm

DEFAULT_ZIP = "/media/pc/ext4TB/geoids_EGM2008 - 1'.zip"
DEFAULT_PARQUET = "/media/pc/HD1/aibirder_model_data/combined.parquet"
DEFAULT_CACHE = Path.home() / ".cache" / "aibirder" / "egm2008"

# EGM2008 1' → raw uint16 → meters.
NGA_SCALE = 0.003
NGA_OFFSET = -108.0


def extract_pgm_from_zip(zip_path: Path, cache_dir: Path) -> Path:
    """Extract the EGM2008 PGM from the zip to a cache dir on first
    call; subsequent calls reuse the cached file. 466 MB — extracting
    once beats streaming on every run."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        pgm_name = next(
            (n for n in zf.namelist() if n.lower().endswith(".pgm")), None
        )
        if pgm_name is None:
            raise RuntimeError(f"No .pgm inside {zip_path}")
        cached = cache_dir / Path(pgm_name).name
        if cached.exists() and cached.stat().st_size > 0:
            return cached
        logging.info("Extracting %s → %s (first run, ~466 MB)", pgm_name, cached)
        with zf.open(pgm_name) as src, open(cached, "wb") as dst:
            while True:
                chunk = src.read(1 << 20)  # 1 MiB
                if not chunk:
                    break
                dst.write(chunk)
        return cached


def read_pgm_header(buf: bytes) -> tuple[int, int, int, int]:
    """Return (width, height, max_val, data_offset). PGM P5 format:
    two-line header with optional `# comment` lines mixed in."""
    if not buf.startswith(b"P5"):
        raise RuntimeError("Not a binary PGM (missing P5 magic).")
    i = 2
    # Consume whitespace + comments between header tokens until we have
    # width, height, maxval (three ints), then exactly one whitespace.
    tokens: list[bytes] = []
    while len(tokens) < 3:
        # skip whitespace
        while i < len(buf) and chr(buf[i]).isspace():
            i += 1
        # skip comment line
        if i < len(buf) and buf[i:i + 1] == b"#":
            while i < len(buf) and buf[i:i + 1] != b"\n":
                i += 1
            continue
        # read token
        j = i
        while j < len(buf) and not chr(buf[j]).isspace():
            j += 1
        tokens.append(buf[i:j])
        i = j
    # single whitespace then data
    if i < len(buf) and chr(buf[i]).isspace():
        i += 1
    w, h, m = (int(t) for t in tokens)
    return w, h, m, i


def load_egm_grid(pgm_path: Path) -> np.ndarray:
    """Memory-map the PGM as uint16 big-endian and return the 2-D
    array. 233 M elements × 2 bytes → memory-mapping avoids hogging
    heap and keeps re-runs fast (OS page cache)."""
    fh = open(pgm_path, "rb")
    mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    # Peek the first few KB for header parsing.
    header_peek = mm[:4096]
    w, h, m, off = read_pgm_header(header_peek)
    if m > 65535:
        raise RuntimeError(f"Unexpected PGM max value {m} — not 16-bit.")
    expected = w * h * 2
    if len(mm) - off < expected:
        raise RuntimeError(
            f"PGM data truncated: expected {expected} bytes after offset {off}, "
            f"got {len(mm) - off}"
        )
    arr = np.frombuffer(mm, dtype=">u2", count=w * h, offset=off).reshape(h, w)
    logging.info("Loaded EGM2008 grid: %d × %d (raw uint16, big-endian)", w, h)
    return arr


def sample_bilinear(grid: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Vectorised bilinear interpolation of the raw uint16 grid at
    (lat, lon) in degrees. Returns raw values (still uint16-valued but
    float). Longitude wraps at 360°; latitude is clamped to the grid
    span (90 S / 90 N). Result is converted to meters by the caller."""
    h, w = grid.shape
    # Grid spacing: height spans 180° across (h-1) intervals; width
    # spans 360° across `w` intervals (column `w` wraps to column 0).
    row = (90.0 - lat) * (h - 1) / 180.0
    lon_mod = np.mod(lon, 360.0)
    col = lon_mod * w / 360.0

    # Clamp row to [0, h-1]; col will wrap naturally.
    row = np.clip(row, 0.0, h - 1)

    r0 = np.floor(row).astype(np.int64)
    r1 = np.minimum(r0 + 1, h - 1)
    dr = row - r0

    c0 = np.floor(col).astype(np.int64) % w
    c1 = (c0 + 1) % w
    dc = col - np.floor(col)

    # Bilinear weights — promote to float32 once, keep memory low.
    v00 = grid[r0, c0].astype(np.float32)
    v01 = grid[r0, c1].astype(np.float32)
    v10 = grid[r1, c0].astype(np.float32)
    v11 = grid[r1, c1].astype(np.float32)
    dr_f = dr.astype(np.float32)
    dc_f = dc.astype(np.float32)
    v0 = v00 * (1.0 - dc_f) + v01 * dc_f
    v1 = v10 * (1.0 - dc_f) + v11 * dc_f
    return v0 * (1.0 - dr_f) + v1 * dr_f


def h3_centroids(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """H3 centroid → (lat, lon). v3 / v4 API-compatible."""
    if hasattr(h3, "cell_to_latlng"):
        fn = h3.cell_to_latlng
    elif hasattr(h3, "h3_to_geo"):
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
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--zip", default=DEFAULT_ZIP,
                        help="EGM2008 1' PGM zip (from NGA).")
    parser.add_argument("--parquet", default=DEFAULT_PARQUET,
                        help="Input observations parquet.")
    parser.add_argument("--out", default=None,
                        help="Output parquet. Defaults to <input>_with_geoid.parquet.")
    parser.add_argument("--h3-col", default="h3_index")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE),
                        help="Where to unpack the PGM on first run.")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--benchmark", action="store_true",
                        help="Skip parquet I/O; just time a 10k-point sample.")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger(__name__)

    zip_path = Path(args.zip)
    if not zip_path.exists():
        raise SystemExit(f"Geoid zip not found: {zip_path}")

    pgm_path = extract_pgm_from_zip(zip_path, Path(args.cache_dir))
    grid = load_egm_grid(pgm_path)

    if args.benchmark:
        import time as _t
        rng = np.random.default_rng(42)
        lat = rng.uniform(-80, 80, size=10_000)
        lon = rng.uniform(-180, 180, size=10_000)
        t0 = _t.time()
        raw = sample_bilinear(grid, lat, lon)
        n = raw * NGA_SCALE + NGA_OFFSET
        dt = _t.time() - t0
        log.info("Bilinear sample of 10k points: %.3f s (%.2f µs/point). N min/max: %.2f / %.2f m",
                 dt, dt * 1e6 / len(lat), n.min(), n.max())
        return

    parquet_path = Path(args.parquet)
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = parquet_path.with_name(parquet_path.stem + "_with_geoid" + parquet_path.suffix)

    log.info("Loading %s", parquet_path)
    df = pd.read_parquet(parquet_path)
    if args.h3_col not in df.columns:
        raise SystemExit(f"Parquet missing column {args.h3_col!r}")

    log.info("Computing H3 centroids for %d rows", len(df))
    indices = df[args.h3_col].to_numpy()
    # Process in chunks to keep a nice progress bar for a 500 k-row parquet.
    chunk = 100_000
    lat = np.empty(len(df), dtype=np.float64)
    lon = np.empty(len(df), dtype=np.float64)
    for start in tqdm(range(0, len(df), chunk), desc="H3 centroids"):
        end = min(start + chunk, len(df))
        la, lo = h3_centroids(indices[start:end])
        lat[start:end] = la
        lon[start:end] = lo

    log.info("Sampling geoid for %d points", len(df))
    raw = sample_bilinear(grid, lat, lon)
    geoid_m = (raw * NGA_SCALE + NGA_OFFSET).astype(np.float32)
    log.info("Geoid range: %.2f to %.2f m (global EGM2008 is ≈ -107 to +86)",
             float(geoid_m.min()), float(geoid_m.max()))

    df["geoid_m"] = geoid_m
    log.info("Writing %s", out_path)
    df.to_parquet(out_path, index=False)
    log.info("Done.")


if __name__ == "__main__":
    main()
