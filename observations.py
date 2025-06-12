import os
import argparse
from tqdm import tqdm
import zipfile
import pandas as pd
from dotenv import load_dotenv
import concurrent.futures
import math

# Load environment variables from .env file
load_dotenv()

# Working directory
WORKING_DIR = os.getenv('WORKING_DIRECTORY', '')

def estimate_zip_contents(file_path, csv_filename, sample_rows=10000):
    """
    Estimate the total number of rows in a zipped CSV file.
    
    Parameters:
    - file_path (str): The path to the zipped CSV file.
    - csv_filename (str): The name of the CSV file inside the zip archive.
    - sample_rows (int): Number of rows to sample for estimation.
    
    Returns:
    - tuple: (estimated_total_rows, estimated_total_size_bytes)
    """
    with zipfile.ZipFile(file_path, 'r') as z:
        info = z.getinfo(csv_filename)
        total_size_bytes = info.file_size
        
        # Read a sample of rows to estimate average row size
        with z.open(csv_filename) as f:
            # Skip header row
            header = f.readline()
            header_size = len(header)
            
            # Read sample rows
            sample_data = b''
            for _ in range(sample_rows):
                line = f.readline()
                if not line:
                    break
                sample_data += line
                
            # Calculate average row size
            if len(sample_data) > 0:
                avg_row_size = len(sample_data) / sample_rows
                # Estimate total rows (accounting for header)
                estimated_total_rows = (total_size_bytes - header_size) / avg_row_size
                return int(estimated_total_rows), total_size_bytes
            else:
                return 0, total_size_bytes

def load_source_observations(file_path, csv_filename, chunk_size=100000, columns=None):
    """
    Load source observations from a zipped CSV file and yield DataFrames in chunks.
    
    Parameters:
    - file_path (str): The path to the zipped CSV file.
    - csv_filename (str): The name of the CSV file inside the zip archive.
    - chunk_size (int): The number of rows to read at a time.
    - columns (list): List of columns to keep. If None, keep all columns.
    
    Yields:
    - pd.DataFrame: A DataFrame containing a chunk of the source observations.
    """
    with zipfile.ZipFile(file_path, 'r') as z:
        with z.open(csv_filename) as f:
            for chunk in pd.read_csv(f, chunksize=chunk_size):
                if columns is not None:
                    chunk = chunk[columns]
                yield chunk

def save_chunk_to_csv(chunk, output_file, mode='a', header=False):
    """
    Save a chunk of DataFrame to a gzipped CSV file.
    
    Parameters:
    - chunk (pd.DataFrame): The DataFrame chunk to save.
    - output_file (str): The path to the output gzipped CSV file.
    - mode (str): The file mode ('a' for append, 'w' for write).
    - header (bool): Whether to write the header row.
    """
    chunk.to_csv(output_file, mode=mode, index=False, header=header, compression='gzip')
                
def process_inat_chunk(chunk):
    """
    Process a single chunk of iNaturalist data.
    """
    # Split occurenceID and make int
    chunk['occurrenceID'] = chunk['occurrenceID'].str.split('/').str[-1].astype(int, errors='ignore')
    
    # Round lat/lon to 3 decimal places
    chunk['decimalLatitude'] = chunk['decimalLatitude'].round(3)
    chunk['decimalLongitude'] = chunk['decimalLongitude'].round(3)
    
    # Split eventDate into date and time
    chunk[['eventDate', 'eventTime']] = chunk['eventDate'].str.split('T', expand=True)
    
    # Save time as hh:mm
    chunk['eventTime'] = chunk['eventTime'].str[:5]    
    
    # Convert column names to lowercase
    chunk.columns = [col.lower() for col in chunk.columns]
    
    return chunk

def parse_inat_source(threads=1):
    """
    Parse iNaturalist source observations and save them to a gzipped CSV file, using multithreading.
    
    Parameters:
    - threads (int): Number of threads to use for processing.
    
    Note: This can take a long time to run, we'll be processing it in chunks to avoid memory issues. The resulting file will be ~2.5GB.
    """    
    
    # Load iNaturalist observations
    # http://www.inaturalist.org/observations/gbif-observations-dwca.zip (Publication date May 27, 2025)
    # Citation: iNaturalist contributors, iNaturalist (2025). iNaturalist Research-grade Observations. iNaturalist.org. Occurrence dataset https://doi.org/10.15468/ab3s5x accessed via GBIF.org on 2025-06-02.
    inat_source_file = f"{WORKING_DIR}/gbif-observations-dwca.zip"
    inat_csv_filename = "observations.csv"
    inat_columns = ['occurrenceID', 'decimalLatitude', 'decimalLongitude', 'eventDate', 'taxonID', 'scientificName']
    inat_dst_file = f"{WORKING_DIR}/inat_parsed_observations.csv.gz"
    
    # Remove existing file if it exists
    if os.path.exists(inat_dst_file):
        os.remove(inat_dst_file)
    
    # Estimate the total number of rows and chunks
    chunk_size = 100000
    estimated_rows, file_size = estimate_zip_contents(inat_source_file, inat_csv_filename)
    estimated_chunks = math.ceil(estimated_rows / chunk_size)
    
    print(f"Processing iNaturalist data: ~{estimated_rows:,} rows ({file_size / (1024*1024*1024):.2f} GB)")
    print(f"Estimated number of chunks: {estimated_chunks:,} with {chunk_size:,} rows per chunk")
    
    # Initialize flags for header and counters
    write_header = True
    processed_chunks = 0
    processed_rows = 0
    
    # Create a generator for chunks
    chunk_generator = load_source_observations(inat_source_file, inat_csv_filename, chunk_size=chunk_size, columns=inat_columns)
    
    # Create global progress bar
    with tqdm(total=estimated_chunks, unit="chunk", desc="Overall iNaturalist progress") as global_pbar:
        # Process in batches
        while True:
            # Create a ThreadPoolExecutor for each batch
            with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
                # Load next N chunks
                futures = []
                chunks_in_batch = 0
                
                # Try to load threads number of chunks
                for _ in range(threads):
                    try:
                        chunk = next(chunk_generator)
                        futures.append(executor.submit(process_inat_chunk, chunk))
                        chunks_in_batch += 1
                    except StopIteration:
                        # No more chunks to process
                        break
                
                # If no chunks were loaded, we're done
                if chunks_in_batch == 0:
                    break
                
                # Process and save the current batch
                with tqdm(total=chunks_in_batch, unit="cnk", desc=f"Batch {processed_chunks//threads + 1}/{math.ceil(estimated_chunks/threads)}") as batch_pbar:
                    for future in concurrent.futures.as_completed(futures):
                        chunk = future.result()
                        rows_in_chunk = len(chunk)
                        processed_rows += rows_in_chunk
                        
                        # Write header only for the first chunk
                        save_chunk_to_csv(chunk, inat_dst_file, mode='a', header=write_header and batch_pbar.n == 0)
                        write_header = False
                        batch_pbar.update(1)
                
                processed_chunks += chunks_in_batch
                global_pbar.update(chunks_in_batch)
                global_pbar.set_postfix(rows=f"{processed_rows:,}/{estimated_rows:,}", percentage=f"{processed_rows/estimated_rows*100:.1f}%")
    
    print(f"Processed {processed_chunks:,} chunks with {processed_rows:,} rows of iNaturalist data")

def process_ebird_chunk(chunk):
    """
    Process a single chunk of eBird data.
    """
    # Parse occurrenceid
    chunk['occurrenceid'] = chunk['occurrenceid'].str.split(':').str[-1].astype(int, errors='ignore')
    
    # Round lat/lon to 3 decimal places
    chunk['decimallatitude'] = chunk['decimallatitude'].round(3)
    chunk['decimallongitude'] = chunk['decimallongitude'].round(3)
    
    # Combine year, month, and day into a single date column (with leading zeros)
    chunk['year'] = chunk['year'].astype(str).str.zfill(4)
    chunk['month'] = chunk['month'].astype(str).str.zfill(2)
    chunk['day'] = chunk['day'].astype(str).str.zfill(2)
    chunk['eventdate'] = chunk[['year', 'month', 'day']].astype(str).agg('-'.join, axis=1)
    chunk.drop(columns=['year', 'month', 'day'], inplace=True)
    
    # Avoid NaN values in individualcount
    chunk['individualcount'] = chunk['individualcount'].fillna(-1).astype(int)
    
    return chunk

def parse_ebird_source(threads=1):
    """
    Parse eBird source observations and save them to a gzipped CSV file, using multithreading.
    
    Parameters:
    - threads (int): Number of threads to use for processing.
    
    Note: This can take a long time to run, we'll be processing it in chunks to avoid memory issues.
    """
    
    # Load eBird observations
    # https://hosted-datasets.gbif.org/eBird/2023-eBird-dwca-1.0.zip (Publication date March 22, 2024)
    # Citation: Auer T, Barker S, Barry J, Charnoky M, Curtis J, Davies I, Davis C, Downie I, Fink D, Fredericks T, Ganger J, Gerbracht J, Hanks C, Hochachka W, Iliff M, Imani J, Jordan A, Levatich T, Ligocki S, Long M T, Morris W, Morrow S, Oldham L, Padilla Obregon F, Robinson O, Rodewald A, Ruiz-Gutierrez V, Schloss M, Smith A, Smith J, Stillman A, Strimas-Mackey M, Sullivan B, Weber D, Wolf H, Wood C (2024). EOD – eBird Observation Dataset. Cornell Lab of Ornithology. Occurrence dataset https://doi.org/10.15468/aomfnb accessed via GBIF.org on 2025-06-02.
    ebird_source_file = f"{WORKING_DIR}/2023-eBird-dwca-1.0.zip"
    ebird_csv_filename = "eod.csv"
    ebird_columns = ['occurrenceid', 'decimallatitude', 'decimallongitude', 'year', 'month', 'day', 'scientificname', 'vernacularname', 'taxonconceptid', 'individualcount']
    ebird_dst_file = f"{WORKING_DIR}/ebird_parsed_observations.csv.gz"
    
    # Remove existing file if it exists
    if os.path.exists(ebird_dst_file):
        os.remove(ebird_dst_file)
    
    # Estimate the total number of rows and chunks
    chunk_size = 100000
    estimated_rows, file_size = estimate_zip_contents(ebird_source_file, ebird_csv_filename)
    estimated_chunks = math.ceil(estimated_rows / chunk_size)
    
    print(f"Processing eBird data: ~{estimated_rows:,} rows ({file_size / (1024*1024*1024):.2f} GB)")
    print(f"Estimated number of chunks: {estimated_chunks:,} with {chunk_size:,} rows per chunk")
    
    # Initialize flags for header and counters
    write_header = True
    processed_chunks = 0
    processed_rows = 0
    
    # Create a generator for chunks
    chunk_generator = load_source_observations(ebird_source_file, ebird_csv_filename, chunk_size=chunk_size, columns=ebird_columns)
    
    # Create global progress bar
    with tqdm(total=estimated_chunks, unit="chunk", desc="Overall eBird progress") as global_pbar:
        # Process in batches
        while True:
            # Create a ThreadPoolExecutor for each batch
            with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
                # Load next N chunks
                futures = []
                chunks_in_batch = 0
                
                # Try to load threads number of chunks
                for _ in range(threads):
                    try:
                        chunk = next(chunk_generator)
                        futures.append(executor.submit(process_ebird_chunk, chunk))
                        chunks_in_batch += 1
                    except StopIteration:
                        # No more chunks to process
                        break
                
                # If no chunks were loaded, we're done
                if chunks_in_batch == 0:
                    break
                
                # Process and save the current batch
                with tqdm(total=chunks_in_batch, unit="cnk", desc=f"Batch {processed_chunks//threads + 1}/{math.ceil(estimated_chunks/threads)}") as batch_pbar:
                    for future in concurrent.futures.as_completed(futures):
                        chunk = future.result()
                        rows_in_chunk = len(chunk)
                        processed_rows += rows_in_chunk
                        
                        # Write header only for the first chunk
                        save_chunk_to_csv(chunk, ebird_dst_file, mode='a', header=write_header and batch_pbar.n == 0)
                        write_header = False
                        batch_pbar.update(1)
                
                processed_chunks += chunks_in_batch
                global_pbar.update(chunks_in_batch)
                global_pbar.set_postfix(rows=f"{processed_rows:,}/{estimated_rows:,}", percentage=f"{processed_rows/estimated_rows*100:.1f}%")
    
    print(f"Processed {processed_chunks:,} chunks with {processed_rows:,} rows of eBird data")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Parse observation data")
    parser.add_argument('--parse_inat_source', action='store_true', help="Parse iNaturalist source observations")
    parser.add_argument('--parse_ebird_source', action='store_true', help="Parse eBird source observations")
    parser.add_argument('--threads', type=int, default=8, help="Number of threads to use for processing (default: 8)")
    
    args = parser.parse_args()
    
    if args.parse_inat_source:
        parse_inat_source(threads=args.threads)
        
    if args.parse_ebird_source:
        parse_ebird_source(threads=args.threads)