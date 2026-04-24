"""
Plot species range maps showing predicted occurrence probability on a global grid.

For each requested species, produces a 2×2 figure with maps for weeks 1, 13,
26, and 39 (roughly Jan, Apr, Jul, Oct).  Grid cells are colored by the
model's predicted probability for that species.

With ``--gif``, all 48 weeks are rendered and combined into an animated GIF.
Multiple species are shown as gridded subplots in each frame.

Usage:
    # By common name (case-insensitive substring match):
    python scripts/plot_range_maps.py --species "Eurasian Blackbird" "House Sparrow"

    # Animated GIF with 4 species in a 2×2 grid:
    python scripts/plot_range_maps.py --species "Barn Swallow" "House Sparrow" \
        "European Robin" "Blue Jay" --gif --cols 2

    # Custom grid resolution and region:
    python scripts/plot_range_maps.py --species "Barn Swallow" --resolution 2.0 --bounds europe

    # With ground truth overlay from training data:
    python scripts/plot_range_maps.py --species "Barn Swallow" --data_path outputs/combined.parquet
"""

import argparse
import io
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch
from PIL import Image

# Add project root to path so we can import project modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.model import create_model
from predict import load_labels
from utils.regions import resolve_bounds_arg

DEFAULT_WEEK_STRIDE = 12

# Weeks excluded from plots by default. Week 48 shares its sin/cos encoding
# with the synthetic "yearly" sample (week=0) inserted by the data loader
# when ``include_yearly=True``. Any model trained with those samples sees
# week 48 as the union-of-all-species target, so its predictions there are
# contaminated. Override with --skip_weeks on the CLI (pass '' to disable).
DEFAULT_SKIP_WEEKS: Tuple[int, ...] = (48,)


def compute_plot_weeks(stride: int, skip: Tuple[int, ...] = DEFAULT_SKIP_WEEKS) -> List[int]:
    """Return the weeks to plot, starting from 1 with the given stride.

    ``stride=12`` → ``[1, 13, 25, 37]`` (4 panels, roughly seasonal).
    ``stride=6``  → ``[1, 7, 13, …, 43]`` (8 panels, bimonthly).
    ``stride=4``  → 12 panels (monthly).

    Any week in ``skip`` is dropped from the output — e.g. to avoid rendering
    week 48, which shares its Fourier encoding with the yearly-bag sample.
    """
    if stride < 1:
        raise ValueError(f'week stride must be ≥ 1, got {stride}')
    skip_set = set(skip or ())
    return [w for w in range(1, 49, stride) if w not in skip_set]


def _layout_rows_cols(n: int) -> Tuple[int, int]:
    """Pick a grid (rows, cols) that's close to square but biased towards
    more columns than rows for better aspect on maps."""
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols

NORMALIZATION_CHOICES = ('raw', 'max', 'sum', 'percentile', 'log')


def apply_normalization(values: np.ndarray, mode: str) -> np.ndarray:
    """Transform raw probabilities according to the selected display mode.

    Normalization is always applied jointly over the values passed in —
    typically all cells × all weeks for a single species — so each species'
    map is comparable with itself across time and space.

    Modes
    -----
    raw
        No transform. Renders the raw sigmoid probability.
    max
        Divide by the maximum value in the pool → [0, 1]. Makes each
        species fill the full colour range regardless of absolute level.
    sum
        *Per-point* normalization: at each (cell, week), divide this
        species' probability by the sum of **all** species' probabilities
        at that same point. Each cell then shows the share of total
        predicted activity that this species represents. Requires the
        full-vocabulary logits, so upstream prediction must produce them.
    percentile
        Rank-transform: each value is mapped to its empirical percentile
        within the pool → [0, 1]. Robust to extreme outliers and makes
        maps across species/sites visually comparable.
    log
        ``log10(value + eps)``. Useful when probabilities span orders of
        magnitude (e.g. well-surveyed lowlands vs sparse uplands).
    """
    if mode == 'raw':
        return values
    arr = np.asarray(values, dtype=float)
    if mode == 'max':
        m = float(arr.max()) if arr.size else 0.0
        return arr / m if m > 0 else arr
    if mode == 'sum':
        # Per-point normalization is handled in predict_grid (requires
        # full-vocab logits); here we just pass the values through.
        return arr
    if mode == 'percentile':
        flat = arr.ravel()
        order = flat.argsort()
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, flat.size + 1)
        return (ranks / flat.size).reshape(arr.shape)
    if mode == 'log':
        eps = 1e-6
        return np.log10(np.clip(arr, eps, None))
    raise ValueError(f'Unknown normalization mode: {mode!r}')


def _colorbar_label(mode: str) -> str:
    return {
        'raw':        'Predicted occurrence probability',
        'max':        'Probability / max (per species)',
        'sum':        'Share of predicted activity at this point',
        'percentile': 'Percentile within species',
        'log':        'log10(probability)',
    }[mode]


def build_grid(
    resolution_deg: float = 2.0,
    bounds: Tuple[float, float, float, float] = (-180.0, -90.0, 180.0, 90.0),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a regular lat/lon grid at the given resolution.

    Returns:
        lats: 1-D array of latitudes
        lons: 1-D array of longitudes
    (as a meshgrid flattened, so each element is one grid cell)
    """
    lon_min, lat_min, lon_max, lat_max = bounds
    lons = np.arange(lon_min + resolution_deg / 2, lon_max, resolution_deg)
    lats = np.arange(lat_min + resolution_deg / 2, lat_max, resolution_deg)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    return lat_grid.ravel(), lon_grid.ravel()


def _add_map_features(ax, *, zorder_bg: int = 0, zorder_fg: int = 3):
    """Add ocean, land, coastlines and borders to a cartopy axis.

    Background features (ocean, land) are drawn at *zorder_bg* and foreground
    line features (coastlines, borders) at *zorder_fg* so they appear on top
    of data layers.
    """
    ax.add_feature(cfeature.OCEAN, facecolor='#e6f0f7', zorder=zorder_bg)
    ax.add_feature(cfeature.LAND, facecolor='#f5f5f5', edgecolor='none', zorder=zorder_bg)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, color='#888888', zorder=zorder_fg)
    ax.add_feature(cfeature.BORDERS, linewidth=0.2, color='#bbbbbb', zorder=zorder_fg)


def _plot_cells(
    ax,
    probs: np.ndarray,
    resolution_deg: float,
    bounds: Tuple[float, float, float, float],
    cmap,
    norm,
):
    """Plot a probability array as gap-free filled cells via *pcolormesh*.

    *probs* is the flat 1-D array produced by :func:`predict_grid`.  It is
    reshaped to the 2-D grid implied by *resolution_deg* and *bounds*.
    Cells with probability <= 0.005 are masked (transparent).
    """
    lon_min, lat_min, lon_max, lat_max = bounds
    n_lons = len(np.arange(lon_min + resolution_deg / 2, lon_max, resolution_deg))
    n_lats = len(np.arange(lat_min + resolution_deg / 2, lat_max, resolution_deg))

    # Cell edges (pcolormesh needs boundaries, not centres)
    lon_edges = np.linspace(lon_min, lon_min + n_lons * resolution_deg, n_lons + 1)
    lat_edges = np.linspace(lat_min, lat_min + n_lats * resolution_deg, n_lats + 1)

    prob_grid = probs.reshape(n_lats, n_lons)
    prob_grid = np.ma.masked_less_equal(prob_grid, 0.005)

    ax.pcolormesh(
        lon_edges, lat_edges, prob_grid,
        cmap=cmap, norm=norm,
        transform=ccrs.PlateCarree(), zorder=2,
    )


def load_ground_truth_data(data_path: str):
    """Load ground truth data from training parquet.

    Returns:
        gt_df: DataFrame with week columns (species lists converted to sets)
        cell_lats: 1-D array of H3 cell latitudes
        cell_lons: 1-D array of H3 cell longitudes
    """
    import h3
    import pandas as pd

    df = pd.read_parquet(data_path)
    coords = np.array([h3.cell_to_latlng(c) for c in df['h3_index'].values])
    cell_lats = coords[:, 0]
    cell_lons = coords[:, 1]

    # Convert week columns to sets for fast membership testing
    for col in [c for c in df.columns if c.startswith('week_')]:
        df[col] = df[col].apply(
            lambda x: {str(s) for s in x} if isinstance(x, (list, np.ndarray)) else set()
        )
    return df, cell_lats, cell_lons


def _plot_gt_cells(ax, gt_df, cell_lats, cell_lons, taxon_key: str, week: int):
    """Overlay ground truth presence dots (green) on a map axis."""
    col = f'week_{week}'
    if col not in gt_df.columns:
        return
    mask = gt_df[col].apply(lambda x: taxon_key in x).values
    if not mask.any():
        return
    ax.scatter(
        cell_lons[mask], cell_lats[mask],
        s=4, c='#2ca02c', marker='o', alpha=0.6,
        transform=ccrs.PlateCarree(), zorder=4,
    )


def load_model_and_labels(checkpoint_path: str, device: torch.device,
                          taxonomy_path: Optional[str] = None):
    """Load model checkpoint and labels.

    Returns ``(model, idx_to_species, labels)``. If ``labels.txt`` is
    present but unnamed (a row's scientific / common name equals the
    taxonKey), names are filled in from ``taxonomy_path`` — auto-detected
    next to the input data parquet or at repo root, or explicit via arg.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = ckpt['model_config']
    species_vocab = ckpt['species_vocab']
    idx_to_species = species_vocab['idx_to_species']

    model = create_model(
        n_species=model_config['n_species'],
        n_env_features=model_config['n_env_features'],
        model_scale=model_config.get('model_scale', 1.0),
        coord_harmonics=model_config.get('coord_harmonics', 8),
        week_harmonics=model_config.get('week_harmonics', 4),
        habitat_head=model_config.get('habitat_head', False),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    ckpt_dir = Path(checkpoint_path).parent
    ckpt_stem = Path(checkpoint_path).stem
    labels_path = ckpt_dir / f'{ckpt_stem}_labels.txt'
    if not labels_path.exists():
        labels_path = ckpt_dir / 'labels.txt'
    labels = load_labels(str(labels_path)) if labels_path.exists() else {}

    # Enrich unnamed labels (where sci == com == taxonKey) from taxonomy CSV.
    labels = _enrich_labels_with_taxonomy(labels, taxonomy_path)
    return model, idx_to_species, labels


def _enrich_labels_with_taxonomy(
    labels: Dict[int, Tuple[str, str, str]],
    taxonomy_path: Optional[str],
) -> Dict[int, Tuple[str, str, str]]:
    """Replace numeric-only label entries with names from the taxonomy CSV."""
    if not labels:
        return labels
    # Nothing to do if every row already has sci != code.
    needs_enrich = any(
        sci == code or com == code
        for code, sci, com in labels.values()
    )
    if not needs_enrich:
        return labels

    candidates: List[Path] = []
    if taxonomy_path:
        candidates.append(Path(taxonomy_path))
    # common locations
    repo_root = Path(__file__).resolve().parent.parent
    candidates.extend([
        repo_root / 'taxonomy.csv',
        Path('/media/pc/HD1/aibirder_model_data/combined_taxonomy.csv'),
        Path('/media/pc/HD1/aibirder_model_data/nordic_combined/taxonomy.csv'),
    ])
    tax_file = next((p for p in candidates if p.is_file()), None)
    if tax_file is None:
        return labels

    import csv as _csv
    tax: Dict[str, Tuple[str, str]] = {}
    with tax_file.open(encoding='utf-8') as f:
        reader = _csv.DictReader(f)
        # support either combine.py-style (taxonKey) or BirdNET-style (species_code)
        for row in reader:
            key = str(row.get('taxonKey') or row.get('species_code') or '').strip()
            if not key:
                continue
            sci = (row.get('sci_name') or row.get('scientificName') or '').strip()
            com = (row.get('com_name') or row.get('commonName') or sci).strip()
            if sci:
                tax[key] = (sci, com)
    if not tax:
        return labels

    out = dict(labels)
    n_enriched = 0
    for idx, (code, sci, com) in labels.items():
        if sci == code or com == code:
            new = tax.get(code)
            if new:
                out[idx] = (code, new[0], new[1])
                n_enriched += 1
    if n_enriched:
        print(f"  Enriched {n_enriched} labels with names from {tax_file}")
    return out


def resolve_species_indices(
    species_names: Optional[List[str]],
    taxon_keys: Optional[List[int]],
    idx_to_species: Dict,
    labels: Dict[int, Tuple[str, str, str]],
) -> List[Tuple[int, str, str, str]]:
    """
    Resolve requested species to model indices.

    Returns list of (model_index, speciesCode, sciName, comName).
    """
    results = []

    if species_names:
        for name_query in species_names:
            query_lower = name_query.lower().strip()
            found = False
            for idx_key, species_id in idx_to_species.items():
                idx = int(idx_key)
                label = labels.get(idx)
                if label:
                    code, sci, com = label
                else:
                    code = sci = com = str(species_id)
                if query_lower in sci.lower() or query_lower in com.lower():
                    if not any(r[0] == idx for r in results):
                        results.append((idx, code, sci, com))
                        found = True
                        break
            if not found:
                print(f"Warning: species '{name_query}' not found in labels, skipping.")

    return results


def predict_grid(
    model: torch.nn.Module,
    lats: np.ndarray,
    lons: np.ndarray,
    week: int,
    species_indices: List[int],
    device: torch.device,
    batch_size: int = 4096,
    point_normalize: bool = False,
) -> np.ndarray:
    """
    Run inference for all grid cells at a given week.

    If ``point_normalize`` is True, the *full* species vector's sigmoid is
    computed per cell and each row is divided by its sum before slicing to
    the requested species. The resulting values are each species' share of
    the total predicted activity at that cell.

    Returns:
        probs: np.ndarray of shape (n_cells, n_requested_species)
    """
    lat_t = torch.from_numpy(lats.astype(np.float32))
    lon_t = torch.from_numpy(lons.astype(np.float32))
    week_t = torch.full((len(lats),), week, dtype=torch.float32)

    all_probs = []
    n = len(lats)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        lb = lat_t[start:end].to(device)
        lnb = lon_t[start:end].to(device)
        wb = week_t[start:end].to(device)
        with torch.no_grad():
            output = model(lb, lnb, wb, return_env=False)
            if point_normalize:
                full = torch.sigmoid(output['species_logits'])
                denom = full.sum(dim=1, keepdim=True).clamp_min(1e-12)
                full = full / denom
                probs = full[:, species_indices].cpu().numpy()
            else:
                logits = output['species_logits'][:, species_indices]
                probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)

    return np.concatenate(all_probs, axis=0)  # (n_cells, n_species)


def plot_range_map(
    lats: np.ndarray,
    lons: np.ndarray,
    probs_per_week: Dict[int, np.ndarray],
    species_info: Tuple[int, int, str, str],
    outdir: str,
    resolution_deg: float,
    bounds: Tuple[float, float, float, float],
    vmax: Optional[float] = None,
    gt_data=None,
    normalization: str = 'raw',
    plot_weeks: Optional[List[int]] = None,
):
    """
    Plot a grid of range maps for one species across the selected weeks.

    Parameters:
        lats, lons: grid cell coordinates
        probs_per_week: {week: probs_array} for each week in *plot_weeks*
        species_info: (model_idx, taxonKey, sciName, comName)
        outdir: output directory
        resolution_deg: grid resolution (for marker sizing)
        bounds: (lon_min, lat_min, lon_max, lat_max)
        vmax: optional max for color scale (default: auto from data)
        normalization: one of NORMALIZATION_CHOICES — applied jointly over
            all selected weeks of this species before rendering.
        plot_weeks: weeks to plot (default: ``compute_plot_weeks(DEFAULT_WEEK_STRIDE)``).
    """
    _, taxon_key, sci_name, com_name = species_info
    is_global = bounds == (-180.0, -90.0, 180.0, 90.0)
    if plot_weeks is None:
        plot_weeks = compute_plot_weeks(DEFAULT_WEEK_STRIDE)

    if normalization != 'raw':
        pooled = np.concatenate([probs_per_week[w] for w in plot_weeks])
        transformed = apply_normalization(pooled, normalization)
        split_sizes = [probs_per_week[w].size for w in plot_weeks]
        offsets = np.cumsum([0] + split_sizes)
        probs_per_week = {
            w: transformed[offsets[i]:offsets[i + 1]]
            for i, w in enumerate(plot_weeks)
        }

    if vmax is None:
        all_vals = np.concatenate([probs_per_week[w] for w in plot_weeks])
        if normalization == 'log':
            vmin = float(np.percentile(all_vals, 5))
            vmax = float(np.percentile(all_vals, 99))
        elif normalization in ('max', 'percentile'):
            vmin, vmax = 0.0, 1.0
        elif normalization == 'sum':
            # Values sum to 1 across cells × weeks; a single cell's share
            # is ≈ 1/N. No 0.05 floor here — that would flatten the scale.
            positive = all_vals[all_vals > 0]
            vmax = float(np.percentile(positive, 99)) if positive.size else 1.0
            vmin = 0.0
        else:
            positive = all_vals[all_vals > 0]
            vmax = float(np.percentile(positive, 99)) if positive.size else 1.0
            vmax = max(vmax, 0.05)
            vmin = 0.0
    else:
        vmin = 0.0 if normalization != 'log' else float(vmax) - 4.0

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.YlOrRd.copy()
    if normalization != 'log':
        cmap.set_under(color='#f0f0f0', alpha=0.0)  # transparent for zero/near-zero

    proj = ccrs.Robinson() if is_global else ccrs.PlateCarree()
    # Suppress cartopy facecolor warning for BORDERS feature
    warnings.filterwarnings('ignore', message='facecolor will have no effect', category=UserWarning)
    nrows, ncols = _layout_rows_cols(len(plot_weeks))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 2.7 + 1.0),
                             subplot_kw=dict(projection=proj), squeeze=False)
    flat_axes = axes.ravel()

    for i, week in enumerate(plot_weeks):
        ax = flat_axes[i]
        probs = probs_per_week[week]

        if is_global:
            ax.set_global()
        else:
            ax.set_extent([bounds[0], bounds[2], bounds[1], bounds[3]], crs=ccrs.PlateCarree())

        _add_map_features(ax)
        _plot_cells(ax, probs, resolution_deg, bounds, cmap, norm)
        if gt_data is not None:
            _plot_gt_cells(ax, *gt_data, taxon_key, week)

        ax.set_title(_week_label(week), fontsize=11, fontweight='bold')

    # Hide any leftover axes when n_weeks doesn't fill the grid
    for ax in flat_axes[len(plot_weeks):]:
        ax.set_visible(False)

    gt_note = "\n(● = observed in training data)" if gt_data is not None else ""
    fig.suptitle(
        f"{com_name} ({sci_name}){gt_note}",
        fontsize=15, fontweight='bold', y=0.98,
    )

    # Shared colorbar
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.04, pad=0.06, shrink=0.6)
    cbar.set_label(_colorbar_label(normalization), fontsize=11)

    os.makedirs(outdir, exist_ok=True)
    safe_name = com_name.replace(' ', '_').replace('/', '_')
    out_path = os.path.join(outdir, f"range_{safe_name}.png")
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# GIF mode
# ---------------------------------------------------------------------------

MONTH_STARTS = {
    1: "Jan", 5: "Feb", 9: "Mar", 13: "Apr", 17: "May", 21: "Jun",
    25: "Jul", 29: "Aug", 33: "Sep", 37: "Oct", 41: "Nov", 45: "Dec",
}


def _week_label(week: int) -> str:
    """Return a human-readable label like 'Week 13 — Apr'."""
    # Find the closest month
    month = "Jan"
    for start_week, name in sorted(MONTH_STARTS.items()):
        if week >= start_week:
            month = name
    return f"Week {week} — {month}"


def _render_gif_frame(
    lats: np.ndarray,
    lons: np.ndarray,
    probs: np.ndarray,
    species_list: List[Tuple[int, int, str, str]],
    week: int,
    resolution_deg: float,
    bounds: Tuple[float, float, float, float],
    n_cols: int,
    vmax_per_species: List[float],
    gt_data=None,
) -> Image.Image:
    """Render a single GIF frame with gridded species subplots.

    Args:
        probs: (n_cells, n_species) probability array.
        species_list: List of (idx, taxonKey, sciName, comName).
        week: Week number.
        n_cols: Number of columns in the subplot grid.
        vmax_per_species: Per-species vmax for consistent color scale.

    Returns:
        PIL Image of the rendered frame.
    """
    n_species = len(species_list)
    n_rows = math.ceil(n_species / n_cols)
    is_global = bounds == (-180.0, -90.0, 180.0, 90.0)

    proj = ccrs.Robinson() if is_global else ccrs.PlateCarree()
    warnings.filterwarnings('ignore', message='facecolor will have no effect', category=UserWarning)

    fig_w = 9 * n_cols
    fig_h = 5 * n_rows + 1.2  # extra space for title + colorbar
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(fig_w, fig_h),
        subplot_kw=dict(projection=proj),
        squeeze=False,
    )

    cmap = plt.cm.YlOrRd.copy()
    cmap.set_under(color='#f0f0f0', alpha=0.0)

    # Global vmax for shared colorbar (max across species)
    global_vmax = max(vmax_per_species) if vmax_per_species else 1.0

    for sp_idx, sp_info in enumerate(species_list):
        row, col = divmod(sp_idx, n_cols)
        ax = axes[row][col]
        _, _, sci_name, com_name = sp_info
        sp_probs = probs[:, sp_idx]
        vmax = vmax_per_species[sp_idx]
        norm = mpl.colors.Normalize(vmin=0.0, vmax=vmax)

        if is_global:
            ax.set_global()
        else:
            ax.set_extent([bounds[0], bounds[2], bounds[1], bounds[3]], crs=ccrs.PlateCarree())

        _add_map_features(ax)
        _plot_cells(ax, sp_probs, resolution_deg, bounds, cmap, norm)
        if gt_data is not None:
            _plot_gt_cells(ax, *gt_data, str(sp_info[1]), week)

        ax.set_title(f"{com_name}", fontsize=11, fontweight='bold')

    # Hide unused subplots
    for idx in range(n_species, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    gt_note = "  (● = observed)" if gt_data is not None else ""
    fig.suptitle(f"{_week_label(week)}{gt_note}", fontsize=16, fontweight='bold', y=0.98)

    # Shared colorbar
    norm = mpl.colors.Normalize(vmin=0.0, vmax=global_vmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal', fraction=0.03, pad=0.04, shrink=0.5)
    cbar.set_label('Predicted occurrence probability', fontsize=11)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    return img


def _plot_gif(
    model: torch.nn.Module,
    lats: np.ndarray,
    lons: np.ndarray,
    model_indices: List[int],
    species_list: List[Tuple[int, int, str, str]],
    device: torch.device,
    batch_size: int,
    resolution_deg: float,
    bounds: Tuple[float, float, float, float],
    outdir: str,
    cols: int,
    fps: int,
    gt_data=None,
    normalization: str = 'raw',
    skip_weeks: Tuple[int, ...] = DEFAULT_SKIP_WEEKS,
):
    """Predict all 48 weeks and assemble an animated GIF."""
    n_species = len(species_list)
    n_cols = cols if cols > 0 else math.ceil(math.sqrt(n_species))
    n_cols = min(n_cols, n_species)  # don't exceed species count

    print(f"GIF mode: {n_species} species in {math.ceil(n_species / n_cols)}×{n_cols} grid, 48 frames")

    # First pass: predict all weeks and compute vmax per species
    skip_set = set(skip_weeks or ())
    all_weeks = [w for w in range(1, 49) if w not in skip_set]
    if skip_set:
        print(f"  (skipping weeks {sorted(skip_set)} from GIF — "
              f"{len(all_weeks)} frames)")
    point_norm = (normalization == 'sum')
    all_probs = {}
    for week in all_weeks:
        print(f"  Predicting week {week}/48...", end='\r')
        all_probs[week] = predict_grid(model, lats, lons, week, model_indices, device, batch_size,
                                       point_normalize=point_norm)
    print(f"  Predictions complete for all 48 weeks.       ")

    # Apply normalization per species, jointly over all 48 weeks
    if normalization != 'raw':
        for sp_idx in range(n_species):
            pooled = np.concatenate([all_probs[w][:, sp_idx] for w in all_weeks])
            transformed = apply_normalization(pooled, normalization)
            n_cells = all_probs[all_weeks[0]].shape[0]
            for i, w in enumerate(all_weeks):
                all_probs[w][:, sp_idx] = transformed[i * n_cells:(i + 1) * n_cells]

    # Compute per-species vmax from the 99th percentile across all weeks
    vmax_per_species = []
    for sp_idx in range(n_species):
        all_vals = np.concatenate([all_probs[w][:, sp_idx] for w in all_weeks])
        if normalization in ('max', 'percentile'):
            vmax = 1.0
        elif normalization == 'log':
            vmax = float(np.percentile(all_vals, 99)) if all_vals.size else 0.0
        elif normalization == 'sum':
            positive = all_vals[all_vals > 0]
            vmax = float(np.percentile(positive, 99)) if positive.size else 1.0
        else:
            positive = all_vals[all_vals > 0]
            if len(positive) > 0:
                vmax = float(np.percentile(positive, 99))
                vmax = max(vmax, 0.05)
            else:
                vmax = 1.0
        vmax_per_species.append(vmax)

    # Render frames
    frames: List[Image.Image] = []
    for week in all_weeks:
        print(f"  Rendering frame {week}/48...", end='\r')
        img = _render_gif_frame(
            lats, lons, all_probs[week], species_list, week,
            resolution_deg, bounds, n_cols, vmax_per_species,
            gt_data=gt_data,
        )
        frames.append(img)
    print(f"  Rendered all 48 frames.                      ")

    # Assemble GIF
    os.makedirs(outdir, exist_ok=True)
    names = '_'.join(info[3].replace(' ', '-') for info in species_list[:4])
    if n_species > 4:
        names += f'_+{n_species - 4}'
    out_path = os.path.join(outdir, f"range_{names}.gif")

    duration_ms = int(1000 / fps)
    frames[0].save(
        out_path, save_all=True, append_images=frames[1:],
        duration=duration_ms, loop=0, optimize=True,
    )
    print(f"Saved {out_path} ({len(frames)} frames, {fps} fps)")


def plot_range_maps(
    species_names: Optional[List[str]] = None,
    checkpoint_path: str = 'checkpoints/checkpoint_best.pt',
    resolution_deg: float = 2.0,
    bounds: Tuple[float, float, float, float] = (-180.0, -90.0, 180.0, 90.0),
    outdir: str = 'outputs/plots/range_maps',
    batch_size: int = 4096,
    device: str = 'auto',
    gif: bool = False,
    cols: int = 0,
    fps: int = 4,
    data_path: Optional[str] = None,
    normalization: str = 'raw',
    week_stride: int = DEFAULT_WEEK_STRIDE,
    skip_weeks: Tuple[int, ...] = DEFAULT_SKIP_WEEKS,
    taxonomy_path: Optional[str] = None,
):
    """Generate species range maps for the given species.

    In default mode, produces a 2×2 seasonal figure per species.
    With *gif=True*, renders all 48 weeks and assembles an animated GIF
    with all species shown in a single gridded figure per frame.
    """
    if not species_names:
        print("Error: provide --species")
        return

    if device == 'auto':
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        dev = torch.device(device)

    print(f"Using device: {dev}")
    model, idx_to_species, labels = load_model_and_labels(
        checkpoint_path, dev, taxonomy_path=taxonomy_path,
    )

    species_list = resolve_species_indices(species_names, None, idx_to_species, labels)
    if not species_list:
        print("No valid species found. Exiting.")
        return

    print(f"Resolved {len(species_list)} species:")
    for _, tk, sci, com in species_list:
        print(f"  {tk}: {com} ({sci})")

    # Build grid
    lats, lons = build_grid(resolution_deg, bounds)
    n_cells = len(lats)
    print(f"Grid: {n_cells} cells at {resolution_deg}° resolution")

    # Collect model indices for batched inference
    model_indices = [sp[0] for sp in species_list]

    # Load ground truth if requested
    gt_data = None
    if data_path:
        print("Loading ground truth data...")
        gt_data = load_ground_truth_data(data_path)
        print(f"Ground truth: {len(gt_data[0])} H3 cells loaded")

    if gif:
        _plot_gif(model, lats, lons, model_indices, species_list,
                  dev, batch_size, resolution_deg, bounds, outdir, cols, fps,
                  gt_data=gt_data, normalization=normalization, skip_weeks=skip_weeks)
    else:
        # Default mode: predict selected weeks, one PNG per species
        plot_weeks = compute_plot_weeks(week_stride, skip=skip_weeks)
        print(f"Plotting {len(plot_weeks)} weeks (stride={week_stride}): {plot_weeks}")
        if skip_weeks:
            print(f"  (skipping weeks {sorted(skip_weeks)} — pass '--skip_weeks' to override)")
        point_norm = (normalization == 'sum')
        probs_by_week = {}
        for week in plot_weeks:
            print(f"  Predicting week {week}...")
            probs = predict_grid(model, lats, lons, week, model_indices, dev, batch_size,
                                 point_normalize=point_norm)
            probs_by_week[week] = probs

        for sp_idx, sp_info in enumerate(species_list):
            sp_probs_per_week = {w: probs_by_week[w][:, sp_idx] for w in plot_weeks}
            plot_range_map(lats, lons, sp_probs_per_week, sp_info, outdir, resolution_deg, bounds,
                           gt_data=gt_data, normalization=normalization,
                           plot_weeks=plot_weeks)

        print(f"\nDone. {len(species_list)} range maps saved to {outdir}/")


def main():
    parser = argparse.ArgumentParser(
        description='Plot species range maps (2×2 seasonal grid) from model predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python scripts/plot_range_maps.py --species "Eurasian Blackbird" "House Sparrow"
  python scripts/plot_range_maps.py --taxon_keys 9750029 9747657
  python scripts/plot_range_maps.py --species "Barn Swallow" --resolution 1.0 --bounds europe
""",
    )
    parser.add_argument('--checkpoint', type=str, default='checkpoints/checkpoint_best.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--species', nargs='+', type=str, default=None,
                        help='Species to plot (common or scientific name, substring match)')
    parser.add_argument('--resolution', type=float, default=2.0,
                        help='Grid resolution in degrees (default: 2.0)')
    parser.add_argument('--bounds', nargs='+', default=['world'],
                        help='Region name (world, europe, usa, ...) or 4 floats: lon_min lat_min lon_max lat_max')
    parser.add_argument('--outdir', type=str, default='outputs/plots/range_maps',
                        help='Output directory for PNGs')
    parser.add_argument('--batch_size', type=int, default=4096,
                        help='Batch size for inference')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--gif', action='store_true',
                        help='Plot all 48 weeks and assemble into an animated GIF')
    parser.add_argument('--cols', type=int, default=0,
                        help='Number of columns for the species grid in GIF mode (default: auto)')
    parser.add_argument('--fps', type=int, default=4,
                        help='Frames per second for the GIF (default: 4)')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to training parquet for ground truth overlay')
    parser.add_argument('--normalization', type=str, default='raw',
                        choices=NORMALIZATION_CHOICES,
                        help='How to scale probabilities for display: '
                             'raw (default, sigmoid values); max (divide by '
                             'per-species max → [0,1]); percentile (rank-'
                             'transform → [0,1], outlier-robust); log '
                             '(log10, good when probs span orders of magnitude).')
    parser.add_argument('--week_stride', type=int, default=DEFAULT_WEEK_STRIDE,
                        help=f'Stride between plotted weeks (default: {DEFAULT_WEEK_STRIDE}). '
                             'Weeks are range(1, 49, stride). 12 → 4 seasonal panels; '
                             '6 → 8 bimonthly; 4 → 12 monthly. Ignored in --gif mode.')
    parser.add_argument('--skip_weeks', type=int, nargs='*',
                        default=list(DEFAULT_SKIP_WEEKS),
                        help='Weeks to exclude from plots. Default: 48 — its sin/cos '
                             'encoding coincides with the synthetic week-0 "yearly" '
                             'sample, so week-48 predictions are contaminated by '
                             'the yearly species bag. Pass without arguments '
                             "(i.e. --skip_weeks) to keep every week.")
    parser.add_argument('--taxonomy', type=str, default=None,
                        help='Path to taxonomy CSV (combined_taxonomy.csv or taxonomy.csv). '
                             'Used to fill in species names when the checkpoint\'s '
                             'labels.txt only contains taxonKeys. Auto-detected at '
                             'repo root and common locations when omitted.')
    args = parser.parse_args()

    bounds = resolve_bounds_arg(args.bounds)
    if bounds is None:
        print(f"Error: could not resolve bounds '{args.bounds}'. Use a region name or 4 floats.")
        sys.exit(1)

    plot_range_maps(
        species_names=args.species,
        checkpoint_path=args.checkpoint,
        resolution_deg=args.resolution,
        bounds=bounds,
        outdir=args.outdir,
        batch_size=args.batch_size,
        device=args.device,
        gif=args.gif,
        cols=args.cols,
        fps=args.fps,
        data_path=args.data_path,
        normalization=args.normalization,
        week_stride=args.week_stride,
        skip_weeks=tuple(args.skip_weeks),
        taxonomy_path=args.taxonomy,
    )


if __name__ == '__main__':
    main()
