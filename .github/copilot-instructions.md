# BirdNET Geomodel Project

## Project Overview
This project builds a spatiotemporal species occurrence prediction model for bird species using H3 geospatial cells and temporal information.

## Data Structure

### Dataset (Parquet files)
Each row represents an H3 cell and contains:
- **h3_index**: H3 geospatial hexagonal grid cell identifier
- **Environmental features**: Multiple geographic/environmental attributes for the cell (elevation, climate data, land cover, etc.)
- **48 week columns** (week_1 through week_48): Each column contains a list of GBIF taxonKeys representing bird species observed in that cell during that specific week of the year

### GBIF TaxonKeys
Species identifiers from the Global Biodiversity Information Facility (GBIF) taxonomy system. Used to represent which bird species were observed.

## Model Architecture & Approach

### Multi-Task Learning Setup

**Model Inputs:**
- **Spatial location**: Latitude and longitude (converted from H3 cells)
  - Uses **sinusoidal encoding** for neural network compatibility
  - Encoding: [sin(lat), cos(lat), sin(lon), cos(lon)] - 4 features
  - Preserves spherical continuity (e.g., -180° and 180° longitude map to same point)
- **Temporal**: Week number (1-48, representing weeks of the year)

**Training Targets:**
- **Primary target**: Species list (GBIF taxonKeys) for that location/week combination
  - Encoded as multi-label binary classification
- **Auxiliary target**: Environmental/geographic features for that cell
  - The environmental features act as a regularization signal
  - Helps the model learn better spatial representations
  - Encourages the model to implicitly capture environmental patterns

**Inference:**
- Only predict species lists for a given location (lat/lon) and week
- Environmental features are NOT used as input or predicted during inference
- The model relies on learned patterns from coordinates and temporal information

### Key Design Principles
1. Environmental data is used during training only (as auxiliary targets, not inputs)
2. The model must learn to encode spatial and temporal patterns from coordinates + week alone
3. Sinusoidal encoding of lat/lon ensures proper handling of geographic coordinates in neural networks
4. Auxiliary environmental prediction helps the model learn meaningful spatial representations
5. At inference time, only (latitude, longitude, week) → species predictions are made

### Technical Implementation Details

**Data Loading:**
- Use GeoPandas to load parquet files (consistent with data creation pipeline)
- H3 cells are converted to lat/lon coordinates using the h3 library
- Each H3 cell × week combination becomes one training sample

**Coordinate Encoding:**
- Latitudes and longitudes are converted to radians
- Applied sinusoidal transformation: [sin(lat_rad), cos(lat_rad), sin(lon_rad), cos(lon_rad)]
- This 4-dimensional encoding preserves spherical geometry and continuity

**Species Encoding:**
- Build vocabulary of all unique GBIF taxonKeys in the dataset
- Convert species lists to multi-label binary format
- Each sample has a binary vector indicating presence/absence of each species

## Project Goals
- Predict which bird species are likely to occur in specific locations (H3 cells) during specific weeks of the year
- Enable species distribution mapping across space and time
- Learn robust spatial-temporal embeddings without requiring environmental data at inference
