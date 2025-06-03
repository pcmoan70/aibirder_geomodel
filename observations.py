import os
import argparse
from tqdm import tqdm
import zipfile
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Working directory
WORKING_DIR = os.getenv('WORKING_DIRECTORY', '')

def load_source_observations(file_path, csv_filename, chunk_size=100000, columns=None):
    """
    Load source observations from a zipped CSV file and yield DataFrames in chunks.
    
    Parameters:
    - file_path (str): The path to the zipped CSV file.
    - chunk_size (int): The number of rows to read at a time.
    
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
                
def save_parsed_observations(df, output_file):
    """
    Save the DataFrame of parsed observations to a gzipped CSV file.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame to save.
    - output_file (str): The path to the output gzipped CSV file.
    """
    df.to_csv(output_file, index=False, compression='gzip')

def parse_inat_source():
    """
    Parse iNaturalist source observations and save them to a gzipped CSV file.
    
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
    
    for chunk in tqdm(load_source_observations(inat_source_file, inat_csv_filename, chunk_size=100000, columns=inat_columns), unit="cnk", desc="Loading iNaturalist observations"):
        
        # Split occurenceID and make int
        chunk['occurrenceID'] = chunk['occurrenceID'].str.split('/').str[-1].astype(int, errors='ignore')
        
        # Round lat/lon to 3 decimal places
        chunk['decimalLatitude'] = chunk['decimalLatitude'].round(3)
        chunk['decimalLongitude'] = chunk['decimalLongitude'].round(3)
        
        # Split eventDate into date and time
        chunk[['eventDate', 'eventTime']] = chunk['eventDate'].str.split('T', expand=True)
        
        # Save time as hh:mm
        chunk['eventTime'] = chunk['eventTime'].str[:5]    
        
        # Convert cloumn names to lowercase
        chunk.columns = [col.lower() for col in chunk.columns]  
            
        # Save chunk to gzipped CSV file
        save_chunk_to_csv(chunk, inat_dst_file, mode='a', header=inat_df.shape[0] == chunk.shape[0])
        
def parse_ebird_source():
    """
    Parse eBird source observations and save them to a gzipped CSV file.
    
    Note: This can take a long time to run, we'll be processing it in chunks to avoid memory issues. The resulting file will be ~XXGB.
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
        
    for chunk in tqdm(load_source_observations(ebird_source_file, ebird_csv_filename, chunk_size=100000, columns=ebird_columns), unit="cnk", desc="Loading eBird observations"):
        
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
        
        # Save chunk to gzipped CSV file
        save_chunk_to_csv(chunk, ebird_dst_file, mode='a', header=chunk.shape[0] == chunk.shape[0])

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Parse observation data")
    parser.add_argument('--parse_inat_source', action='store_true', help="Parse iNaturalist source observations")
    parser.add_argument('--parse_ebird_source', action='store_true', help="Parse eBird source observations")
    
    args = parser.parse_args()
    
    if args.parse_inat_source:
        parse_inat_source()
        
    if args.parse_ebird_source:
        parse_ebird_source()