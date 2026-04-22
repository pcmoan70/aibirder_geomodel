# Geoid models are used with GPS to convert ellipsoidal heights (h, based on the WGS84 ellipsoid) 
# to orthometric heights (H, elevation above sea level) by subtracting the geoid height (N), calculated as 
# H=h-N
# These models represent Earth's gravity field (mean sea level) to determine true elevation.


import pygeodesy
import time
from pygeodesy.ellipsoidalKarney import LatLon
ginterpolator = pygeodesy.GeoidKarney("EGM2008-1.pgm")

# Make an example location
lat=51.416422
lon=-116.217151

lat=89.9 
lon=-70.054429

# Get the geoid height
print(100*"-")
C = 10000
start_t = time.time()
for i in range(C):
	single_position=LatLon(lat, lon)
	N = ginterpolator(single_position)
end_t = time.time()
print(h, "time:", (end_t-start_t)/C)
