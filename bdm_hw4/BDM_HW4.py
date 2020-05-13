
from pyspark import SparkContext
import sys

yellow = sys.argv[1]
output_path = sys.argv[2]


if __name__=='__main__':
	sc = SparkContext()

def createIndex(shapefile):
    '''
    This function takes in a shapefile path, and return:
    (1) index: an R-Tree based on the geometry data in the file
    (2) zones: the original data of the shapefile
    
    Note that the ID used in the R-tree 'index' is the same as
    the order of the object in zones.
    '''
    import rtree
    import fiona.crs
    import geopandas as gpd
    zones = gpd.read_file(shapefile).to_crs(fiona.crs.from_epsg(2263))
    index = rtree.Rtree()
    for idx,geometry in enumerate(zones.geometry):
        index.insert(idx, geometry.bounds)
    return (index, zones)

def findZone(p, index, zones):
    '''
    findZone returned the ID of the shape (stored in 'zones' with
    'index') that contains the given point 'p'. If there's no match,
    None will be returned.
    '''
    match = index.intersection((p.x, p.y, p.x, p.y))
    for idx in match:
        if zones.geometry[idx].contains(p):
            return idx
    return None

def processTrips(pid, records):
    '''
    Our aggregation function that iterates through records in each
    partition, checking whether we could find a zone that contain
    the pickup location.
    '''
    import csv
    import pyproj
    import shapely.geometry as geom
    
    # Create an R-tree index
    proj = pyproj.Proj(init="epsg:2263", preserve_units=True)    
    index, zones = createIndex('neighborhoods.geojson')    
    
    # Skip the header
    if pid==0:
        next(records)
    reader = csv.reader(records)
    counts = {}
    
    for row in reader:
        try:
            
            # (longitude, latitude)
            p_pickup = geom.Point(proj(float(row[5]), float(row[6]))) 
            p_dropoff = geom.Point(proj(float(row[9]), float(row[10])))     

            # Look up a matching zone, and update the count accordly if
            # such a match is found

            # get index of zone information 
            zone_pickup = findZone(p_pickup, index, zones)
            zone_dropoff = findZone(p_dropoff, index, zones)

            if zone_pickup and zone_dropoff:
            # get zone info from index
                pickup_bor = zones.iloc[zone_pickup,:]['borough']
                dropoff_nei = zones.iloc[zone_dropoff,:]['neighborhood']
            # make trip info dictionary entry, count
                trip = (pickup_bor, dropoff_nei)
                counts[trip] = counts.get(trip, 0) + 1 
        except:
            print('Error occurred.')
            
    return counts.items()

rdd = sc.textFile(yellow)
counts = rdd.mapPartitionsWithIndex(processTrips) \
    .reduceByKey(lambda x,y: x+y) \
    .map(lambda x: (x[0][0], [(x[0][1], x[1])])) \
    .reduceByKey(lambda x,y: x+y) \
    .mapValues(lambda x: sorted(x, key=lambda tup: tup[1], reverse=True)[:3]) \
    .sortByKey() \
    .map(lambda x: (str(x[0])+','+ str(x[1][0][0])+','+str(x[1][0][1])+','+str(x[1][1][0])+','+str(x[1][1][1])+','+str(x[1][2][0])+','+str(x[1][2][1])))

counts.collect()
counts.saveAsTextFile(output_path)