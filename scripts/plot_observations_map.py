"""Plot observation cell centroids from a combined GeoParquet on a folium map.

The combined parquet stores observations aggregated per H3 cell × week.
This script keeps only cells that have at least one observation in any week
and plots their centroids as clustered markers.

Usage::

    python scripts/plot_observations_map.py \\
        --input /media/pc/HD1/aibirder_model_data/combined.parquet \\
        --output outputs/plots/observations_map.html
"""

import argparse
from pathlib import Path

import folium
import geopandas as gpd
import numpy as np
from folium.plugins import FastMarkerCluster

DEFAULT_OUTPUT_NAME = 'observations_map.html'


def resolve_output_path(arg: str) -> Path:
    """Normalize --output: accepts a directory, a bare name, or a full path.

    - Trailing slash or existing directory → append the default filename.
    - Missing extension → append .html.
    - Bare name (no separator) → place under outputs/plots/.
    """
    raw = arg.strip()
    looks_like_dir = raw.endswith(('/', '\\'))
    p = Path(raw)
    if looks_like_dir or (p.exists() and p.is_dir()):
        p = p / DEFAULT_OUTPUT_NAME
    if p.suffix == '':
        p = p.with_suffix('.html')
    if p.parent == Path(''):
        p = Path('outputs/plots') / p
    return p


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--input', '-i', required=True, help='Combined GeoParquet path')
    parser.add_argument('--output', '-o', default='outputs/plots/observations_map.html')
    args = parser.parse_args()

    print(f'Loading {args.input} ...')
    gdf = gpd.read_parquet(args.input)
    print(f'  {len(gdf):,} cells')

    week_cols = [c for c in gdf.columns if c.startswith('week_')]
    has_obs = np.zeros(len(gdf), dtype=bool)
    for c in week_cols:
        has_obs |= gdf[c].apply(lambda a: a is not None and len(a) > 0).to_numpy()
    gdf = gdf.loc[has_obs]
    print(f'  {len(gdf):,} cells with at least one observation')

    b = gdf.geometry.bounds
    lon = ((b['minx'] + b['maxx']) * 0.5).to_numpy()
    lat = ((b['miny'] + b['maxy']) * 0.5).to_numpy()
    coords = np.column_stack([lat, lon])
    coords = np.unique(coords, axis=0)
    print(f'  {len(coords):,} unique centroids')
    points = coords.tolist()

    center = [float(lat.mean()), float(lon.mean())]
    fmap = folium.Map(location=center, zoom_start=5, tiles='cartodbpositron')
    FastMarkerCluster(points).add_to(fmap)

    out_path = resolve_output_path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out_path))
    print(f'Saved {out_path}')


if __name__ == '__main__':
    main()
