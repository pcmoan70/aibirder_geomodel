"""Combine H3 geodata with processed GBIF observations.

Maps each GBIF record to its H3 cell and week, producing a combined
parquet with per-week species lists and an accompanying taxonomy CSV.
"""

import geopandas as gpd
import gzip
import logging
import os
from collections import defaultdict

import h3
import pandas as pd
from tqdm import tqdm

# Columns read from the processed GBIF CSV
GBIF_REQUIRED_COLUMNS = ['latitude', 'longitude', 'taxonKey', 'verbatimScientificName', 'commonName', 'week', 'class']

# Number of BirdNET weeks per year
NUM_WEEKS = 48


def estimate_gzip_rows(file_path, sample_rows=10000):
    """Estimate total rows in a gzipped CSV by sampling compressed byte ratios."""
    compressed_size = os.path.getsize(file_path)

    with gzip.open(file_path, 'rb') as f:
        f.readline()  # skip header
        start_pos = f.fileobj.tell()

        for _ in range(sample_rows):
            if not f.readline():
                break

        end_pos = f.fileobj.tell()

    compressed_sample_size = end_pos - start_pos
    if compressed_sample_size > 0:
        return max(int((compressed_size - start_pos) / (compressed_sample_size / sample_rows)), 1)
    return 0


def combine_geodata_and_gbif(geodata_path, gbif_processed_path, output_path, valid_classes):
    """Combine geographical data with processed GBIF data.

    Reads an H3-indexed GeoParquet and a processed GBIF CSV, maps each GBIF
    observation to its H3 cell and week, and writes a combined parquet with
    per-week species lists.

    Args:
        geodata_path (str): Path to the H3-indexed GeoParquet file.
        gbif_processed_path (str): Path to the processed GBIF CSV (.csv.gz).
        output_path (str): Output path for the combined parquet.
        valid_classes (list[str]): Taxonomic classes to include.
    """
    gdf = gpd.read_parquet(geodata_path)

    # Determine H3 resolution from data
    h3_res = h3.get_resolution(gdf.iloc[0]['h3_index'])
    logging.info(f"H3 resolution {h3_res}")

    valid_h3_cells = set(gdf['h3_index'])
    valid_classes_set = set(valid_classes)
    missing_cells = set()

    # Accumulate species per (cell, week) — sets for automatic deduplication
    cell_week_species = defaultdict(set)

    # Collect taxonKey → (scientificName, commonName) mapping
    taxon_names: dict = {}

    estimated_rows = estimate_gzip_rows(gbif_processed_path)

    # Pandas handles gzip natively — no need for manual gzip.open
    reader = pd.read_csv(
        gbif_processed_path, chunksize=500_000, usecols=GBIF_REQUIRED_COLUMNS,
    )
    with tqdm(total=estimated_rows, desc="Processing GBIF data") as pbar:
        for chunk in reader:
            chunk_size = len(chunk)

            # Filter to valid classes
            chunk = chunk[chunk['class'].isin(valid_classes_set)]

            if not chunk.empty:
                # Compute H3 cells
                h3_cells = [
                    h3.latlng_to_cell(lat, lon, h3_res)
                    for lat, lon in zip(chunk['latitude'].values,
                                        chunk['longitude'].values)
                ]
                chunk = chunk.assign(h3_cell=h3_cells)

                # Track missing cells (log once per cell)
                mask = chunk['h3_cell'].isin(valid_h3_cells)
                new_missing = set(chunk.loc[~mask, 'h3_cell'].unique()) - missing_cells
                if new_missing:
                    logging.warning(f"{len(new_missing)} new H3 cell(s) not in geodata")
                    missing_cells.update(new_missing)

                matched = chunk.loc[mask]
                if not matched.empty:
                    # Accumulate species per (cell, week) via groupby
                    for (cell, week), species in matched.groupby(
                        ['h3_cell', 'week'],
                    )['taxonKey'].agg(set).items():
                        cell_week_species[(cell, int(week))] |= species

                    # Batch-collect taxon names (once per unique taxonKey)
                    taxa = matched.drop_duplicates('taxonKey')
                    for tk, sci, com in zip(
                        taxa['taxonKey'].values,
                        taxa['verbatimScientificName'].values,
                        taxa['commonName'].values,
                    ):
                        taxon_names.setdefault(int(tk), (str(sci), str(com)))

            pbar.update(chunk_size)

    # Build week columns efficiently: construct plain lists, assign once
    # (avoids slow per-cell gdf.at[] calls)
    h3_to_idx = dict(zip(gdf['h3_index'], gdf.index))
    n_rows = len(gdf)

    week_lists = {w: [[] for _ in range(n_rows)] for w in range(1, NUM_WEEKS + 1)}
    for (h3_cell, week), species_set in cell_week_species.items():
        week_lists[week][h3_to_idx[h3_cell]] = list(species_set)

    for week in range(1, NUM_WEEKS + 1):
        gdf[f'week_{week}'] = week_lists[week]

    if missing_cells:
        logging.info(f"Total {len(missing_cells)} H3 cell(s) from GBIF not found in geodata")

    gdf.to_parquet(output_path, index=False)
    logging.info(f"Saved combined dataset to {output_path}")

    # Save taxonomy CSV (taxonKey, scientificName, commonName) alongside the parquet
    taxonomy_path = output_path.replace('.parquet', '_taxonomy.csv') if isinstance(output_path, str) else str(output_path).replace('.parquet', '_taxonomy.csv')
    taxonomy_df = pd.DataFrame([
        {'taxonKey': k, 'scientificName': sci, 'commonName': com}
        for k, (sci, com) in sorted(taxon_names.items())
    ])
    taxonomy_df.to_csv(taxonomy_path, index=False)
    logging.info(f"Saved taxonomy ({len(taxon_names)} species) to {taxonomy_path}")


if __name__ == '__main__':
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Combine geographical data with processed GBIF data.')
    parser.add_argument('--geodata', type=str, default='./data/global_350km_ee.parquet',
                        help='Path to the geographical data parquet file')
    parser.add_argument('--gbif', type=str, default='./outputs/gbif_processed.gz',
                        help='Path to the gzipped processed GBIF file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output parquet file. If not provided, generated from geodata filename.')
    parser.add_argument('--valid_classes', type=str, nargs='+', default=['Aves', 'Mammalia', 'Amphibia'],
                        help='Valid taxonomic classes to include from GBIF data')
    args = parser.parse_args()

    output = args.output
    if output is None:
        geodata_filename = os.path.basename(args.geodata).replace('.parquet', '_gbif.parquet')
        output = os.path.join('./outputs', geodata_filename)
        logging.info(f"No output file provided. Using default: {output}")

    combine_geodata_and_gbif(args.geodata, args.gbif, output, args.valid_classes)