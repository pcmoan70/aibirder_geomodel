"""
Plot species range maps from the BirdNET+ Geomodel V3 TFLite build.

Sister of ``plot_range_maps.py``. Where the original reaches into the
PyTorch checkpoint, this one loads the pre-exported TFLite file from
``geomodelV3/``:

    geomodelV3/BirdNET+_Geomodel_V3.0.2_Global_12K_FP16.tflite
    geomodelV3/BirdNET+_Geomodel_V3.0.2_Global_12K_Labels.txt

The model input is ``[lat, lon, raw_week]`` (shape ``[1, 3]`` float32);
the output is a 12 012-dim vector with one per-species score per cell.
Weeks use the raw 1..48 convention — same as the legacy BN v2 meta-model.

Usage
-----
    python scripts/plot_range_maps_v3.py --species "Eurasian Blackbird"
    python scripts/plot_range_maps_v3.py --species "Barn Swallow" --bounds europe
    python scripts/plot_range_maps_v3.py --species "House Sparrow" "Blue Jay" \
        --gif --cols 2

The FlexOps set embedded in the model (Erf is one of them) requires a
TensorFlow Python build that bundles the Flex delegate. The standard
``tensorflow`` or ``ai_edge_litert`` wheel supplies it on most
platforms; if Python complains with ``FlexErf failed to prepare`` pin
TF ≥ 2.15 + Flex ops (``pip install tensorflow``).

Many plumbing pieces — grid building, figure layout, normalization
choices, GIF assembly — are reused verbatim from ``plot_range_maps.py``.
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
from PIL import Image

# Reuse the shared helpers from plot_range_maps so both scripts stay in
# lockstep on grid layout, normalization, colouring and GIF assembly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.plot_range_maps import (  # noqa: E402
    DEFAULT_SKIP_WEEKS,
    DEFAULT_WEEK_STRIDE,
    NORMALIZATION_CHOICES,
    _add_map_features,
    _layout_rows_cols,
    _plot_cells,
    _plot_gt_cells,
    _week_label,
    _colorbar_label,
    apply_normalization,
    build_grid,
    compute_plot_weeks,
    load_ground_truth_data,
)
from utils.regions import resolve_bounds_arg  # noqa: E402

DEFAULT_MODEL_PATH = str(Path(__file__).resolve().parent.parent
                         / 'geomodelV3'
                         / 'BirdNET+_Geomodel_V3.0.2_Global_12K_FP16.tflite')
DEFAULT_LABELS_PATH = str(Path(__file__).resolve().parent.parent
                          / 'geomodelV3'
                          / 'BirdNET+_Geomodel_V3.0.2_Global_12K_Labels.txt')


# ---------------------------------------------------------------------------
# Inference backend
# ---------------------------------------------------------------------------
#
# The V3 TFLite build embeds FlexOps (Erf etc.) that the standard
# `tensorflow` wheel on PyPI no longer ships with its Flex delegate.
# Converting the TFLite once into ONNX via tf2onnx sidesteps the issue:
# the resulting graph runs on plain onnxruntime (no Flex, no Android
# toolchain), and the numerical output is identical. We auto-convert on
# first use and cache the `.onnx` next to the `.tflite`.


class _Backend:
    """Thin wrapper so `predict_grid_v3` doesn't care whether we're on
    ONNX Runtime or TFLite underneath."""

    def __init__(self, run_fn, name: str):
        self.run = run_fn
        self.name = name


def load_backend(model_path: str) -> _Backend:
    """Return a callable that evaluates a batched ``[N, 3]`` input.

    Prefers ONNX Runtime (no Flex delegate needed). Falls back to
    `tf.lite.Interpreter`; that path requires a TF build with the Flex
    delegate (`FlexErf`) — absent in stock TF 2.21 PyPI wheels.
    """
    p = Path(model_path)
    onnx_path = p.with_suffix('.onnx')
    if not onnx_path.exists() and p.suffix.lower() == '.tflite':
        print(f"  First-time conversion: {p.name} → {onnx_path.name} (via tf2onnx)...")
        _convert_tflite_to_onnx(str(p), str(onnx_path))

    if onnx_path.exists():
        try:
            import onnxruntime as ort  # type: ignore
            sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
            in_name = sess.get_inputs()[0].name
            def run(x: np.ndarray) -> np.ndarray:
                return sess.run(None, {in_name: x.astype(np.float32)})[0]
            return _Backend(run, f'onnxruntime ({onnx_path.name})')
        except ImportError:
            print('  onnxruntime not installed; falling back to TFLite.')

    # TFLite fallback — only works if the user has a TF build with Flex.
    import tensorflow as tf  # type: ignore
    interp = tf.lite.Interpreter(model_path=str(p))
    interp.allocate_tensors()
    in_idx = interp.get_input_details()[0]['index']
    out_idx = interp.get_output_details()[0]['index']
    current_batch = [0]

    def run(x: np.ndarray) -> np.ndarray:
        actual = x.shape[0]
        if actual != current_batch[0]:
            interp.resize_tensor_input(in_idx, (actual, 3))
            interp.allocate_tensors()
            current_batch[0] = actual
        interp.set_tensor(in_idx, x.astype(np.float32))
        interp.invoke()
        return interp.get_tensor(out_idx)

    return _Backend(run, f'tflite ({p.name})')


def _convert_tflite_to_onnx(tflite_path: str, onnx_path: str) -> None:
    """Convert the TFLite model to ONNX via tf2onnx's CLI entry point.

    Using the CLI avoids picking an unstable subset of the Python API
    across tf2onnx versions; ~1-2 s one-shot conversion, cached thereafter.
    """
    try:
        from tf2onnx import convert  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            'tf2onnx not installed — either install it (`pip install tf2onnx`) '
            'so the V3 model can be converted to ONNX, or install a TF build '
            'that bundles the Flex delegate (needed for FlexErf in this model).'
        ) from e
    import subprocess
    subprocess.check_call([
        sys.executable, '-m', 'tf2onnx.convert',
        '--tflite', tflite_path,
        '--output', onnx_path,
    ])


# ---------------------------------------------------------------------------
# Labels + interpreter
# ---------------------------------------------------------------------------


def load_v3_labels(path: str) -> List[Tuple[str, str, str]]:
    """Load the V3 label file (TSV: taxonKey, sci, com)."""
    rows: List[Tuple[str, str, str]] = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) < 3:
                # Some pipelines emit `sci_underscore_com` in a single field.
                if len(parts) == 1 and '_' in parts[0]:
                    code = ''
                    sci_com = parts[0]
                    if ' ' in sci_com:
                        sci, com = sci_com.split(' ', 1)
                    else:
                        sci = com = sci_com.replace('_', ' ')
                    rows.append((code, sci, com))
                    continue
                raise ValueError(f'Bad label line (expected taxonKey\\tsci\\tcom): {line!r}')
            rows.append((parts[0], parts[1], parts[2]))
    return rows


def resolve_species_indices_v3(
    species_names: Optional[List[str]],
    labels: List[Tuple[str, str, str]],
) -> List[Tuple[int, str, str, str]]:
    """Match user queries (substring on sci or com) to label rows.

    Returns ``[(model_index, taxonKey, sci, com), ...]``.
    """
    results: List[Tuple[int, str, str, str]] = []
    if not species_names:
        return results
    for q in species_names:
        ql = q.lower().strip()
        for idx, (code, sci, com) in enumerate(labels):
            if ql in sci.lower() or ql in com.lower():
                if not any(r[0] == idx for r in results):
                    results.append((idx, code, sci, com))
                break
        else:
            print(f"Warning: species '{q}' not found in labels, skipping.")
    return results


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def predict_grid_v3(
    backend: _Backend,
    lats: np.ndarray,
    lons: np.ndarray,
    week: int,
    species_indices: List[int],
    batch_size: int = 1024,
    point_normalize: bool = False,
) -> np.ndarray:
    """Run V3 meta-model inference for all grid cells at *week*.

    The backend consumes ``[N, 3]`` float32 inputs — whichever runtime
    underneath (ONNX Runtime or TFLite) handles batching internally.

    ``point_normalize=True`` divides each cell's full-vocab output by
    its sum before slicing to the requested species — each cell then
    shows each species' share of total predicted activity. The V3 head
    already emits probability-range values (max ≈ 0.87 in practice), so
    no extra sigmoid is applied.
    """
    n = len(lats)
    species_idx = np.asarray(species_indices, dtype=np.int64)
    out = np.empty((n, len(species_indices)), dtype=np.float32)

    lats32 = lats.astype(np.float32)
    lons32 = lons.astype(np.float32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        inp = np.empty((end - start, 3), dtype=np.float32)
        inp[:, 0] = lats32[start:end]
        inp[:, 1] = lons32[start:end]
        inp[:, 2] = float(week)
        y = backend.run(inp)  # (actual, n_classes)
        if point_normalize:
            denom = y.sum(axis=1, keepdims=True)
            denom[denom < 1e-12] = 1e-12
            out[start:end] = (y / denom)[:, species_idx]
        else:
            out[start:end] = y[:, species_idx]

    return out


# ---------------------------------------------------------------------------
# Plotting (delegates to helpers from plot_range_maps)
# ---------------------------------------------------------------------------


def plot_range_map_v3(
    lats: np.ndarray,
    lons: np.ndarray,
    probs_per_week: Dict[int, np.ndarray],
    species_info: Tuple[int, str, str, str],
    outdir: str,
    resolution_deg: float,
    bounds: Tuple[float, float, float, float],
    vmax: Optional[float] = None,
    gt_data=None,
    normalization: str = 'raw',
    plot_weeks: Optional[List[int]] = None,
):
    """Render one 2×2 (or N-panel) range map per species, V3 model.

    Same visual layout as ``plot_range_maps.plot_range_map``; only the
    probability source differs. Re-uses the normalisation / colour /
    feature helpers so renderings stay consistent.
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
        cmap.set_under(color='#f0f0f0', alpha=0.0)

    proj = ccrs.Robinson() if is_global else ccrs.PlateCarree()
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

    for ax in flat_axes[len(plot_weeks):]:
        ax.set_visible(False)

    gt_note = "\n(● = observed in training data)" if gt_data is not None else ""
    fig.suptitle(f"{com_name} ({sci_name}){gt_note}  [V3]",
                 fontsize=15, fontweight='bold', y=0.98)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal',
                        fraction=0.04, pad=0.06, shrink=0.6)
    cbar.set_label(_colorbar_label(normalization), fontsize=11)

    os.makedirs(outdir, exist_ok=True)
    safe_name = com_name.replace(' ', '_').replace('/', '_')
    out_path = os.path.join(outdir, f"range_v3_{safe_name}.png")
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved {out_path}")


# ---------------------------------------------------------------------------
# GIF mode
# ---------------------------------------------------------------------------


def _render_gif_frame_v3(
    lats: np.ndarray,
    lons: np.ndarray,
    probs: np.ndarray,
    species_list: List[Tuple[int, str, str, str]],
    week: int,
    resolution_deg: float,
    bounds: Tuple[float, float, float, float],
    n_cols: int,
    vmax_per_species: List[float],
    gt_data=None,
) -> Image.Image:
    """One frame of the all-species / all-weeks GIF."""
    n_species = len(species_list)
    n_rows = math.ceil(n_species / n_cols)
    is_global = bounds == (-180.0, -90.0, 180.0, 90.0)

    proj = ccrs.Robinson() if is_global else ccrs.PlateCarree()
    warnings.filterwarnings('ignore', message='facecolor will have no effect', category=UserWarning)

    fig_w = 9 * n_cols
    fig_h = 5 * n_rows + 1.2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h),
                             subplot_kw=dict(projection=proj), squeeze=False)

    cmap = plt.cm.YlOrRd.copy()
    cmap.set_under(color='#f0f0f0', alpha=0.0)

    global_vmax = max(vmax_per_species) if vmax_per_species else 1.0

    for sp_idx, sp_info in enumerate(species_list):
        row, col = divmod(sp_idx, n_cols)
        ax = axes[row][col]
        _, taxon_key, sci_name, com_name = sp_info
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
            _plot_gt_cells(ax, *gt_data, taxon_key, week)

        ax.set_title(f"{com_name}", fontsize=11, fontweight='bold')

    for idx in range(n_species, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    gt_note = "  (● = observed)" if gt_data is not None else ""
    fig.suptitle(f"{_week_label(week)}{gt_note}  [V3]",
                 fontsize=16, fontweight='bold', y=0.98)

    norm = mpl.colors.Normalize(vmin=0.0, vmax=global_vmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=axes, orientation='horizontal',
                        fraction=0.03, pad=0.04, shrink=0.5)
    cbar.set_label('Predicted occurrence probability', fontsize=11)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert('RGB')


def _plot_gif_v3(
    backend: _Backend,
    lats: np.ndarray,
    lons: np.ndarray,
    model_indices: List[int],
    species_list: List[Tuple[int, str, str, str]],
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
    """Same structure as the V1 GIF mode — predict all 48 weeks, render frames."""
    n_species = len(species_list)
    n_cols = cols if cols > 0 else math.ceil(math.sqrt(n_species))
    n_cols = min(n_cols, n_species)

    print(f"GIF mode [V3]: {n_species} species in {math.ceil(n_species / n_cols)}×{n_cols} grid")

    skip_set = set(skip_weeks or ())
    all_weeks = [w for w in range(1, 49) if w not in skip_set]
    if skip_set:
        print(f"  (skipping weeks {sorted(skip_set)} — {len(all_weeks)} frames)")
    point_norm = (normalization == 'sum')
    all_probs: Dict[int, np.ndarray] = {}
    for week in all_weeks:
        print(f"  Predicting week {week}/48...", end='\r')
        all_probs[week] = predict_grid_v3(backend, lats, lons, week, model_indices,
                                          batch_size, point_normalize=point_norm)
    print(f"  Predictions complete for {len(all_weeks)} weeks.       ")

    if normalization != 'raw':
        for sp_idx in range(n_species):
            pooled = np.concatenate([all_probs[w][:, sp_idx] for w in all_weeks])
            transformed = apply_normalization(pooled, normalization)
            n_cells = all_probs[all_weeks[0]].shape[0]
            for i, w in enumerate(all_weeks):
                all_probs[w][:, sp_idx] = transformed[i * n_cells:(i + 1) * n_cells]

    vmax_per_species: List[float] = []
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
            vmax = float(np.percentile(positive, 99)) if positive.size else 1.0
            vmax = max(vmax, 0.05)
        vmax_per_species.append(vmax)

    frames: List[Image.Image] = []
    for week in all_weeks:
        print(f"  Rendering frame {week}/48...", end='\r')
        img = _render_gif_frame_v3(
            lats, lons, all_probs[week], species_list, week,
            resolution_deg, bounds, n_cols, vmax_per_species,
            gt_data=gt_data,
        )
        frames.append(img)
    print(f"  Rendered {len(frames)} frames.                      ")

    os.makedirs(outdir, exist_ok=True)
    names = '_'.join(info[3].replace(' ', '-') for info in species_list[:4])
    if n_species > 4:
        names += f'_+{n_species - 4}'
    out_path = os.path.join(outdir, f"range_v3_{names}.gif")
    duration_ms = int(1000 / fps)
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=duration_ms, loop=0, optimize=True)
    print(f"Saved {out_path} ({len(frames)} frames, {fps} fps)")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def plot_range_maps_v3(
    species_names: Optional[List[str]] = None,
    model_path: str = DEFAULT_MODEL_PATH,
    labels_path: str = DEFAULT_LABELS_PATH,
    resolution_deg: float = 2.0,
    bounds: Tuple[float, float, float, float] = (-180.0, -90.0, 180.0, 90.0),
    outdir: str = 'outputs/plots/range_maps_v3',
    batch_size: int = 1024,
    gif: bool = False,
    cols: int = 0,
    fps: int = 4,
    data_path: Optional[str] = None,
    normalization: str = 'raw',
    week_stride: int = DEFAULT_WEEK_STRIDE,
    skip_weeks: Tuple[int, ...] = DEFAULT_SKIP_WEEKS,
):
    if not species_names:
        print("Error: provide --species")
        return

    print(f"Loading labels: {labels_path}")
    labels = load_v3_labels(labels_path)
    print(f"  {len(labels)} labels")

    species_list = resolve_species_indices_v3(species_names, labels)
    if not species_list:
        print("No valid species found. Exiting.")
        return
    print(f"Resolved {len(species_list)} species:")
    for _, tk, sci, com in species_list:
        print(f"  {tk}: {com} ({sci})")

    print(f"Loading model: {model_path}")
    backend = load_backend(model_path)
    print(f"  backend: {backend.name}")

    lats, lons = build_grid(resolution_deg, bounds)
    print(f"Grid: {len(lats)} cells at {resolution_deg}° resolution")

    model_indices = [sp[0] for sp in species_list]

    gt_data = None
    if data_path:
        print("Loading ground truth data...")
        gt_data = load_ground_truth_data(data_path)
        print(f"Ground truth: {len(gt_data[0])} H3 cells loaded")

    if gif:
        _plot_gif_v3(backend, lats, lons, model_indices, species_list,
                     batch_size, resolution_deg, bounds, outdir, cols, fps,
                     gt_data=gt_data, normalization=normalization,
                     skip_weeks=skip_weeks)
    else:
        plot_weeks = compute_plot_weeks(week_stride, skip=skip_weeks)
        print(f"Plotting {len(plot_weeks)} weeks (stride={week_stride}): {plot_weeks}")
        if skip_weeks:
            print(f"  (skipping weeks {sorted(skip_weeks)} — pass '--skip_weeks' to override)")
        point_norm = (normalization == 'sum')
        probs_by_week: Dict[int, np.ndarray] = {}
        for week in plot_weeks:
            print(f"  Predicting week {week}...")
            probs_by_week[week] = predict_grid_v3(
                backend, lats, lons, week, model_indices,
                batch_size, point_normalize=point_norm,
            )

        for sp_idx, sp_info in enumerate(species_list):
            sp_probs_per_week = {w: probs_by_week[w][:, sp_idx] for w in plot_weeks}
            plot_range_map_v3(lats, lons, sp_probs_per_week, sp_info, outdir,
                              resolution_deg, bounds,
                              gt_data=gt_data, normalization=normalization,
                              plot_weeks=plot_weeks)

        print(f"\nDone. {len(species_list)} V3 range maps saved to {outdir}/")


def main():
    parser = argparse.ArgumentParser(
        description='Plot species range maps from BirdNET+ Geomodel V3 (TFLite)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python scripts/plot_range_maps_v3.py --species "Eurasian Blackbird"
  python scripts/plot_range_maps_v3.py --species "Barn Swallow" --resolution 1.0 --bounds europe
  python scripts/plot_range_maps_v3.py --species "House Sparrow" "Blue Jay" --gif --cols 2
""",
    )
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH,
                        help='Path to the V3 TFLite file')
    parser.add_argument('--labels', type=str, default=DEFAULT_LABELS_PATH,
                        help='Path to the V3 labels file (tsv: taxonKey\\tsci\\tcom)')
    parser.add_argument('--species', nargs='+', type=str, default=None,
                        help='Species to plot (common or scientific, substring match)')
    parser.add_argument('--resolution', type=float, default=2.0,
                        help='Grid resolution in degrees (default: 2.0)')
    parser.add_argument('--bounds', nargs='+', default=['world'],
                        help='Region name (world, europe, usa, ...) or 4 floats: lon_min lat_min lon_max lat_max')
    parser.add_argument('--outdir', type=str, default='outputs/plots/range_maps_v3',
                        help='Output directory for PNGs / GIF')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size for TFLite inference (default: 1024)')
    parser.add_argument('--gif', action='store_true',
                        help='Render all 48 weeks and assemble an animated GIF')
    parser.add_argument('--cols', type=int, default=0,
                        help='Columns for species grid in GIF mode (default: auto)')
    parser.add_argument('--fps', type=int, default=4,
                        help='GIF frames per second (default: 4)')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Training parquet for ground-truth overlay')
    parser.add_argument('--normalization', type=str, default='raw',
                        choices=NORMALIZATION_CHOICES,
                        help='How to scale probabilities for display')
    parser.add_argument('--week_stride', type=int, default=DEFAULT_WEEK_STRIDE,
                        help=f'Stride between plotted weeks (default: {DEFAULT_WEEK_STRIDE})')
    parser.add_argument('--skip_weeks', type=int, nargs='*',
                        default=list(DEFAULT_SKIP_WEEKS),
                        help='Weeks to exclude (default: 48).')
    args = parser.parse_args()

    bounds = resolve_bounds_arg(args.bounds)
    if bounds is None:
        print(f"Error: could not resolve bounds '{args.bounds}'.")
        sys.exit(1)

    plot_range_maps_v3(
        species_names=args.species,
        model_path=args.model,
        labels_path=args.labels,
        resolution_deg=args.resolution,
        bounds=bounds,
        outdir=args.outdir,
        batch_size=args.batch_size,
        gif=args.gif,
        cols=args.cols,
        fps=args.fps,
        data_path=args.data_path,
        normalization=args.normalization,
        week_stride=args.week_stride,
        skip_weeks=tuple(args.skip_weeks),
    )


if __name__ == '__main__':
    main()
