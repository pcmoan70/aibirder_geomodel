import os
import argparse
import math
import ee
import pandas as pd
from tqdm import tqdm
from global_land_mask import globe
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from matplotlib.patches import Rectangle
import random
import matplotlib.cm as cm
from dotenv import load_dotenv

load_dotenv()

# Load working directory from environment variable
WORKING_DIR = os.getenv('WORKING_DIRECTORY', '')

# Random seed for reproducibility
random.seed(42)

# Initialize the Earth Engine API
ee.Initialize()

def is_ocean(lat, lon):
    """
    Check if a given location is in the ocean/sea or on land using the global_land_mask package.
    
    Parameters:
    - lat: Latitude of the location
    - lon: Longitude of the location
    
    Returns:
    - Boolean: True if the location is in ocean/sea, False if it's on land
    """
    return not globe.is_land(lat, lon)

def generate_geo_grid(start_lat=-90, end_lat=90, grid_step_km=100, ocean_sample_chance=0.1):
    """
    Generate a grid of latitudes and longitudes with a given grid size.
    Only adds land points to the grid. Ocean points are added with a 10% chance.
    
    Parameters:
    - start_lat: The starting latitude (-90 to 90).
    - end_lat: The ending latitude (-90 to 90).
    - grid_step_km: The step size for the grid in kilometers.
    - ocean_sample_chance: Probability of sampling an ocean point (default: 0.1).
    
    Returns:
    - Generator yielding (latitude, longitude, is_ocean) tuples
    """
    # Earth's radius in km (approximate)
    earth_radius = 6371.0
    
    # Calculate latitude step in degrees
    lat_step_deg = (grid_step_km / earth_radius) * (180 / math.pi)

    lat = start_lat
    while lat <= end_lat:
        lon = -180
        while lon <= 180:
            # Determine if the current location is ocean or land
            is_ocean_value = is_ocean(lat, lon)
            
            # Add land points to the grid. Ocean points are added with a 10% chance.
            if (not is_ocean_value and lat > -60 and lat < 75) or random.random() < ocean_sample_chance:
                yield round(lat, 3), round(lon, 3), is_ocean_value
            
            # Calculate longitude step in degrees at this latitude
            cos_lat = math.cos(math.radians(lat))
            if abs(cos_lat) < 0.01:
                lon_step_deg = 10.0
            else:
                lon_step_deg = (grid_step_km / earth_radius) * (180 / math.pi) / cos_lat
            
            lon += lon_step_deg
        
        lat += lat_step_deg

def get_landcover_class_name(class_value):
    """
    Map MODIS MCD12Q1 IGBP land cover class values to their descriptive names.
    
    Parameters:
    - class_value: Numeric class value from the MODIS land cover dataset
    
    Returns:
    - String description of the land cover class
    """
    landcover_classes = {
        0: "Water",
        1: "Evergreen Needleleaf Forest",
        2: "Evergreen Broadleaf Forest",
        3: "Deciduous Needleleaf Forest",
        4: "Deciduous Broadleaf Forest",
        5: "Mixed Forest",
        6: "Closed Shrublands",
        7: "Open Shrublands",
        8: "Woody Savannas",
        9: "Savannas",
        10: "Grasslands",
        11: "Permanent Wetlands",
        12: "Croplands",
        13: "Urban and Built-up",
        14: "Cropland/Natural Vegetation Mosaic",
        15: "Snow and Ice",
        16: "Barren or Sparsely Vegetated",
        254: "Unclassified",
        255: "Fill Value"
    }
    
    return landcover_classes.get(class_value, f"Unknown Class ({class_value})")

def get_landcover_data(lat, lon):
    """
    Retrieve the basic land cover data for a given latitude and longitude using Google Earth Engine.
    
    Parameters:
    - lat: Latitude of the location
    - lon: Longitude of the location
    
    Returns:
    - Tuple (landcover_class, landcover_name) or (None, "Unknown") if data not available
    """
    try:
        point = ee.Geometry.Point(lon, lat)
        landcover = ee.ImageCollection("MODIS/061/MCD12Q1").select('LC_Type1')
        landcover_image = landcover.filterDate('2023-01-01', '2023-12-31').mean()
        landcover_value = landcover_image.sample(region=point, scale=1000, numPixels=1).first()
        
        if landcover_value is None:
            return None, "Unknown"
            
        landcover_class = landcover_value.get('LC_Type1').getInfo()
        landcover_name = get_landcover_class_name(landcover_class)
        
        return int(landcover_class), landcover_name
        
    except Exception as e:
        return None, "Unknown"

def get_elevation(lat, lon):
    """
    Get elevation data for a given latitude and longitude.
    
    Parameters:
    - lat: Latitude of the location
    - lon: Longitude of the location
    
    Returns:
    - Elevation in meters above sea level
    """
    try:
        point = ee.Geometry.Point(lon, lat)
        elevation = ee.Image("USGS/SRTMGL1_003")
        elevation_value = elevation.sample(region=point, scale=100, numPixels=1).first()
        return elevation_value.get("elevation").getInfo()
    
    except Exception as e:
        return 0

def get_tree_canopy_height(lat, lon):
    """
    Get tree canopy height for a given latitude and longitude.
    
    Parameters:
    - lat: Latitude of the location
    - lon: Longitude of the location
    
    Returns:
    - Tree canopy height in meters
    """
    try:
        point = ee.Geometry.Point(lon, lat)
        canopy = ee.Image("NASA/JPL/global_forest_canopy_height_2005")
        canopy_value = canopy.sample(region=point, scale=100, numPixels=1).first()
        return canopy_value.get("1").getInfo()
    
    except Exception as e:
        return 0

def get_climate_data(lat, lon, is_ocean=False):
    """
    Get climate data (temperature and precipitation) for a given latitude and longitude.
    
    Parameters:
    - lat: Latitude of the location
    - lon: Longitude of the location
    - is_ocean: Boolean indicating if the location is in the ocean
    
    Returns:
    - Dictionary with annual mean temperature and precipitation
    """
    point = ee.Geometry.Point(lon, lat)
    
    try:
    
        if is_ocean:        
            oisst = ee.ImageCollection("NOAA/CDR/SST_WHOI/V2")
            sst = oisst.filterDate('2023-01-01', '2023-12-31').mean().select('sea_surface_temperature')
            climate_values = sst.sample(region=point, scale=1000, numPixels=1).first()
            temp = round(climate_values.get("sea_surface_temperature").getInfo(), 1)
            precip = None  # Precipitation data not available for ocean locations
        else:
            worldclim = ee.Image("WORLDCLIM/V1/BIO")
            climate_values = worldclim.sample(region=point, scale=1000, numPixels=1).first()
            temp = round(climate_values.get("bio01").getInfo() / 10, 1)
            precip = int(climate_values.get("bio12").getInfo())
            
    except Exception as e:
        #print(f"Error retrieving climate data for ({lat}, {lon}): {e}")
        temp = None
        precip = None
    
    return {
        "temperature": temp,
        "precipitation": precip
    }

def get_environmental_data(lat, lon, is_ocean):
    """
    Get comprehensive environmental data for a given latitude and longitude.
    
    Parameters:
    - lat: Latitude of the location
    - lon: Longitude of the location
    - is_ocean: Boolean indicating if the location is in the ocean
    
    Returns:
    - Dictionary with various environmental parameters
    """
    climate = get_climate_data(lat, lon, is_ocean=is_ocean)
    
    if is_ocean:
        landcover_class = 0
        landcover_string = "Ocean/Sea"
        elevation = 0
        canopy_height = 0
    else:
        landcover_class, landcover_string = get_landcover_data(lat, lon)
        elevation = get_elevation(lat, lon)
        canopy_height = get_tree_canopy_height(lat, lon)
    
    return {
        "latitude": lat,
        "longitude": lon,
        "is_ocean": is_ocean,
        "landcover_class": int(landcover_class) if landcover_class is not None else None,
        "landcover_name": landcover_string,
        "elevation_m": elevation,
        "canopy_height_m": canopy_height,
        "temperature_c": climate["temperature"],
        "precipitation_mm": climate["precipitation"]
    }
    
def get_environmental_data_batch(grid_step_km=100, start_lat=-85, end_lat=85, ocean_sample_chance=0.1):
    """
    Get environmental data for a grid of locations across the Earth.
    
    Parameters:
    - grid_step_km: Step size for the grid in kilometers.
    - start_lat: Starting latitude (default: -85 to avoid extreme poles)
    - end_lat: Ending latitude (default: 85 to avoid extreme poles)
    - ocean_sample_chance: Probability of sampling an ocean point (default: 0.1).
    
    Returns:
    - List of dictionaries containing environmental data for each location
    """
    data = []
    grid_gen = generate_geo_grid(
        start_lat=start_lat, 
        end_lat=end_lat,
        grid_step_km=grid_step_km,
        ocean_sample_chance=ocean_sample_chance
    )
    
    data = [get_environmental_data(lat, lon, is_ocean) for lat, lon, is_ocean in tqdm(list(grid_gen), desc="Processing locations")]
    
    return data
    
def save_environmental_data_to_csv(data, filename):
    """
    Save environmental data to a CSV file.
    
    Parameters:
    - data: List of dictionaries containing environmental data
    - filename: Name of the output CSV file
    """
    df = pd.DataFrame(data)
    df['landcover_class'] = df['landcover_class'].astype('Int64')
    df['precipitation_mm'] = df['precipitation_mm'].astype('Int64')
    df.to_csv(filename, index=False)
    print(f"Environmental data saved to {filename}")

def plot_column_values(grid_data, column_name, grid_step_km, cmap='viridis'):
    """
    Plots the values from a specified column of the grid data on a map.

    Parameters:
    - grid_data: A Pandas DataFrame containing the grid data.
                   Must have 'latitude', 'longitude', and the specified column.
    - column_name: The name of the column to plot.
    - grid_step_km: Step size for the grid in kilometers (used for rectangle dimensions).
    - cmap: Colormap to use for the plot (default: 'viridis').
    """

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # Add continent outlines
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)

    # Function to calculate the bounding box for a given lat, lon, and step_km
    def calculate_bbox(lat, lon, step_km):
        earth_radius = 6371.0
        lat_step_deg = (step_km / earth_radius) * (180 / math.pi)
        cos_lat = math.cos(math.radians(lat))
        if abs(cos_lat) < 0.01:
            lon_step_deg = 10.0
        else:
            lon_step_deg = (step_km / earth_radius) * (180 / math.pi) / cos_lat
        
        lower_lon = lon - lon_step_deg / 2.0
        lower_lat = lat - lat_step_deg / 2.0
        upper_lon = lon + lon_step_deg / 2.0
        upper_lat = lat + lat_step_deg / 2.0
        return [lower_lon, lower_lat, upper_lon, upper_lat]

    # Find min and max values for the color scale
    min_val = grid_data[column_name].min()
    max_val = grid_data[column_name].max()

    # Plot grid cells as rectangles, colored by the column value
    for index, row in grid_data.iterrows():
        lat = row['latitude']
        lon = row['longitude']
        value = row[column_name]

        bbox = calculate_bbox(lat, lon, grid_step_km)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]

        # Normalize the value to be between 0 and 1 for the colormap
        normalized_value = (value - min_val) / (max_val - min_val)

        # Get the color from the colormap
        color = plt.colormaps.get_cmap(cmap)(normalized_value)

        rect = Rectangle((bbox[0], bbox[1]), width, height,
                         facecolor=color, edgecolor='none', alpha=0.7,
                         transform=ccrs.PlateCarree())
        ax.add_patch(rect)

    ax.set_global()
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Grid Data: {column_name}")

    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_val, vmax=max_val))
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', shrink=0.7)
    cbar.set_label(column_name)

    plt.savefig(f"plots/geo_grid_plot_{column_name}_{grid_step_km}km.png", bbox_inches='tight', dpi=300)
    plt.show()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Generate a grid of environmental data.")
    parser.add_argument('--grid_step_km', type=int, default=100, help='Step size for the grid in kilometers (default: 100)')
    parser.add_argument('--ocean_sample_chance', type=float, default=0.01, help='Probability of sampling an ocean point (default: 0.01)')
    parser.add_argument('--plot_column', type=str, default='elevation_m', help='Column to plot (default: elevation_m). None to skip plotting.')
    args = parser.parse_args()
    
    grid_step_km = args.grid_step_km
    ocean_sample_chance = args.ocean_sample_chance
    filepath = f"{WORKING_DIR}/environmental_data_{grid_step_km}km.csv"
    
    if not os.path.exists(filepath):    
        # Get environmental data for the grid
        environmental_data = get_environmental_data_batch(grid_step_km=grid_step_km, ocean_sample_chance=ocean_sample_chance)
        save_environmental_data_to_csv(environmental_data, filename=filepath)
    else:           
        # Load the grid data from a CSV file
        grid_data = pd.read_csv(filepath)
        print(f"Loaded {len(grid_data)} grid points from CSV.")
    
    # Plot the grid
    if args.plot_column and args.plot_column in grid_data.columns:
        print(f"Plotting column '{args.plot_column}' with step size {grid_step_km} km...")
        plot_column_values(grid_data, args.plot_column, grid_step_km, cmap='viridis')
        print(f"...Done! Plot saved as 'plots/geo_grid_plot_{args.plot_column}_{grid_step_km}km.png'")        
    else:
        print(f"Column '{args.plot_column}' not found in the grid data. Skipping plotting.")