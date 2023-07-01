import rasterio
import math
import os
import numpy as np
import pandas as pd
import geopandas as gp
from shapely.geometry import box
from shapely.geometry import shape
from rasterio.features import shapes
import subprocess

def coordinates(raster):
    height = raster.height
    width = raster.width
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rasterio.transform.xy(raster.transform, rows, cols)
    xcoords = np.array(xs)
    ycoords = np.array(ys)
    return xcoords, ycoords

def numpy_coordinates(idx_narry, value):
    nrows_ncolumns = np.argwhere(idx_narry == value)
    # output is (y,x)
    return nrows_ncolumns[0,0], nrows_ncolumns[0,1]

def clump(segmented_image, reference_transform=None, connectivity=8, maskValue=0):
    mask = segmented_image!=maskValue

    featureGenerator = shapes(segmented_image,
                              transform = reference_transform,
                              connectivity = connectivity,
                              mask=mask)
    return featureGenerator

def createGDFfromShapes(featureGenerator, fieldName='rasterVal'):
    geomList = []
    rasterValList = []

    for feat in featureGenerator:
        geomList.append(shape(feat[0]))
        rasterValList.append(feat[1])

    featureGDF = gp.GeoDataFrame({'geometry': geomList, fieldName: rasterValList})

    return featureGDF

def distance_polys_table(polygons_gp, dropzeros = False, decimals = 2):
    distance_matrix = polygons_gp.geometry.apply(lambda g: polygons_gp.distance(g))
    distance_matrix = np.around(distance_matrix, decimals=decimals)

    # Into long table format.
    distance_matrix = distance_matrix.where(np.triu(np.ones(distance_matrix.shape),
                                                    k=0).astype(bool)).stack().reset_index()

    distance_matrix.iloc[:,0:2] = distance_matrix.iloc[:, 0:2] + 1
    distance_matrix = distance_matrix.query("level_0 == level_1")
    if dropzeros:
        distance_matrix = distance_matrix.drop(distance_matrix[distance_matrix[0] == 0].index)
    return distance_matrix

def distance_polys_matrix(polygons_gp, probability=0.5, dispersal_distance=2000, decimals = 5):
    distance_matrix = polygons_gp.geometry.apply(lambda g: polygons_gp.distance(g))
    distance_matrix = distance_matrix.to_numpy()
    k = math.log(probability) / dispersal_distance
    distance_matrix = np.exp(k * distance_matrix)
    np.fill_diagonal(distance_matrix, 0)
    distance_matrix = np.around(distance_matrix, decimals=decimals)
    return distance_matrix

def area_matrix(polygons_layer, outer = False):
    if outer:
        area_matrix = np.outer(polygons_layer.area, polygons_layer.area)
    else:
        area_matrix = polygons_layer.area
        area_matrix = np.vstack((np.arange(1,area_matrix.shape[0]+1),area_matrix.to_numpy())).T
    return area_matrix

def extent_area(layer, crs):
    extent_bounds = layer.geometry.total_bounds
    extent_box = box(extent_bounds[0], extent_bounds[1], extent_bounds[2], extent_bounds[3])
    gdf = gp.GeoDataFrame(index=[0], crs=crs, geometry=[extent_box])
    return gdf.area[0]

def shortest_paths(adjacency_matrix):
    netwerk = nx.from_numpy_array(adjacency_matrix, create_using=nx.Graph)
    paths = dict(nx.all_pairs_dijkstra_path_length(netwerk, weight='weight'))
    paths_df = pd.DataFrame.from_dict(paths).sort_index()
    return paths_df

def pc(paths_df, area_matrix, extent_area = None, decimals = 4):
    path_areas = area_matrix * paths_df.to_numpy()
    pc = path_areas.sum() / extent_area ** 2
    pc = int(np.floor(pc*(100)))
    return pc

def conefor(polygons_area_matrix,
            path,
            nodes_file = "nodes.txt",
            distances_file = "distances.txt",
            conefor_executable = "coneforWin64.exe",
            ec=False):

    CONEFOR_EXECUTABLE = path + conefor_executable
    NODES_FILE = path + nodes_file
    DISTANCES_FILE = path + distances_file

    index = 0

    if polygons_area_matrix.shape[0] == 1:
        index = polygons_area_matrix[0,1]*polygons_area_matrix[0,1]

    else:
        result = subprocess.call([CONEFOR_EXECUTABLE,
                              "-nodeFile", NODES_FILE,
                              "-conFile", DISTANCES_FILE,
                              "-confAdj", "85",
                              "-IIC", "onlyoverall"],
                             stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)

        indices = np.loadtxt('overall_indices.txt', usecols=(1))

        if ec:
            index = indices[1]
        else:
            index = indices[0]

    # clean up
    if os.path.isfile(NODES_FILE):
        os.remove(NODES_FILE)
    else:
        pass
    if os.path.isfile(DISTANCES_FILE):
        os.remove(DISTANCES_FILE)
    else:
        pass
    if os.path.isfile('overall_indices.txt'):
        os.remove('overall_indices.txt')
    else:
        pass
    if os.path.isfile('results_all_EC(IIC).txt'):
        os.remove('results_all_EC(IIC).txt')
    else:
        pass
    return index