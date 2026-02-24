import logging
import zipfile
import numpy as np
import pandas as pd
from tqdm import tqdm


# Output CSV column order
OUTPUT_COLUMNS = ['latitude', 'longitude', 'taxonKey', 'verbatimScientificName', 'week', 'class']

# Required source columns (must all be non-null)
REQUIRED_COLUMNS = ['decimalLatitude', 'decimalLongitude', 'day', 'month', 'taxonKey', 'verbatimScientificName', 'class']


def date_to_week(day, month):
    """
    Convert day/month arrays to BirdNET week numbers (1-48, 4 weeks per month).
    Works with both scalar and vectorized (numpy array) inputs.
    """
    week = (np.asarray(month, dtype=int) - 1) * 4
    day = np.asarray(day, dtype=int)
    week = week + np.where(day <= 7, 1, np.where(day <= 14, 2, np.where(day <= 21, 3, 4)))
    return week


def estimate_rows(zip_archive, file_path, sample_rows=10000):
    """
    Estimate the total number of rows in a zipped CSV file by sampling.
    """
    total_size_bytes = zip_archive.getinfo(file_path).file_size

    with zip_archive.open(file_path) as f:
        header = f.readline()
        header_size = len(header.decode())

        sample_data = b''
        for _ in range(sample_rows):
            line = f.readline()
            if not line:
                break
            sample_data += line

    if sample_data:
        avg_row_size = len(sample_data) / sample_rows
        estimated_total_rows = (total_size_bytes - header_size) / avg_row_size
        return max(int(estimated_total_rows), 1)
    return 0


def process_gbif_file(gbif_zip_path, file, output_csv_path, valid_classes=None, max_rows=None):
    """
    Process a GBIF Darwin Core Archive zip, filter and transform records,
    and write the result to a (optionally gzipped) CSV.
    """
    # Write CSV header
    pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(output_csv_path, index=False, encoding='utf-8')

    valid_classes_set = set(valid_classes) if valid_classes else None
    rows_processed = 0

    with zipfile.ZipFile(gbif_zip_path, 'r') as z:
        estimated_rows = estimate_rows(z, file)
        with z.open(file) as f:
            with tqdm(total=estimated_rows, desc="Processing GBIF data") as pbar:
                for chunk in pd.read_csv(f, sep='\t', chunksize=100000, on_bad_lines='warn',
                                        usecols=REQUIRED_COLUMNS):
                    chunk_size = len(chunk)

                    # Drop rows missing any required field
                    chunk = chunk.dropna(subset=REQUIRED_COLUMNS)

                    # Filter to valid taxonomic classes
                    if valid_classes_set and not chunk.empty:
                        chunk = chunk[chunk['class'].str.lower().isin(valid_classes_set)]

                    # Keep only full species (1-2 word names, skip subspecies / higher taxa)
                    if not chunk.empty:
                        chunk = chunk[chunk['verbatimScientificName'].str.split().str.len() <= 2]

                    if not chunk.empty:
                        chunk = chunk.copy()
                        chunk['latitude'] = chunk['decimalLatitude'].astype(float).round(3)
                        chunk['longitude'] = chunk['decimalLongitude'].astype(float).round(3)
                        chunk = chunk.dropna(subset=['latitude', 'longitude'])

                    if not chunk.empty:
                        chunk['week'] = date_to_week(chunk['day'], chunk['month'])
                        chunk[OUTPUT_COLUMNS].to_csv(
                            output_csv_path, mode='a', header=False, index=False, encoding='utf-8'
                        )

                    pbar.update(chunk_size)
                    rows_processed += chunk_size
                    if max_rows is not None and rows_processed >= max_rows:
                        break


if __name__ == '__main__':
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='Process the zipped GBIF file and extract relevant columns.')
    parser.add_argument('--gbif', type=str, default="gbif_dev.zip", help='Path to the zipped GBIF file')
    parser.add_argument('--file', type=str, default="gbif_dev.csv", help='Name of the CSV file inside the zip')
    parser.add_argument('--output', type=str, default="./outputs/gbif_processed.gz", help='Output gzipped CSV file')
    parser.add_argument('--valid_classes', nargs='*', default=['aves', 'amphibia', 'insecta', 'mammalia', 'reptilia'],
                        help='List of classes to include (default: aves, amphibia, insecta, mammalia, reptilia)')
    parser.add_argument('--max_rows', type=int, default=None, help='Maximum number of rows to process (for testing)')
    args = parser.parse_args()

    process_gbif_file(
        args.gbif, args.file, args.output,
        valid_classes=[cls.lower() for cls in args.valid_classes],
        max_rows=args.max_rows,
    )