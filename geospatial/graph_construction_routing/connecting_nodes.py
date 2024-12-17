import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

import rasterio as rio
from rasterio.plot import show
import rioxarray as rxr
from rasterio.plot import plotting_extent

import geopandas as gpd
from geopandas import GeoSeries
import osmnx as ox
from osgeo import gdal, osr

from shapely.geometry import mapping
from shapely.geometry import box
from shapely.geometry import Point, LineString
from shapely.geometry.polygon import Polygon
import shapely


import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep

from skimage.graph import route_through_array

import networkx as nx
import momepy

import fiona

import pandas as pd

from osgeo import  ogr, gdal, osr, os
import numpy as np
import itertools
from math import sqrt,ceil
from shapely.geometry import shape as shp

import richdem as rd
import folium

import xarray as xr

from xrspatial import a_star_search
from xrspatial.utils import get_dataarray_resolution

import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import geopandas as gpd

import warnings
from typing import Optional, Union

import numpy as np
import xarray as xr

from xrspatial.utils import get_dataarray_resolution, ngjit

from tqdm.notebook import trange, tqdm
import pickle
from joblib import Parallel, delayed


# def nearest_node_loop(nodes_lst, G):
#     nearest_lst = [] 
#     node_name = len(G.nodes) +1
    
#     for i in nodes_lst:
#         dist= ox.distance.nearest_nodes( G, i["x"], i["y"])
#         nearest_lst.append({"x_start" : i["x"], "y_start": i["y"],  "x_end": G.nodes[dist]['x'], "y_end" : G.nodes[dist]['y'], "v" : dist, "u": node_name })
#         node_name+=1
#     return nearest_lst

def raster2array(rasterfn):
    raster = gdal.Open(rasterfn)
    band = raster.GetRasterBand(1)
    array = band.ReadAsArray()
    # print(array)
    array = array[:-1, 1:]
    return array

def coord2pixelOffset(rasterfn,x,y):
    raster = gdal.Open(rasterfn)
    geotransform = raster.GetGeoTransform()
    # print(geotransform)
    originX = geotransform[0]
    originY = geotransform[3]
    pixelWidth = geotransform[1]
    pixelHeight = geotransform[5]
    xOffset = int((x - originX)/pixelWidth)
    yOffset = int((y - originY)/pixelHeight)
    # print(xffset, yOffset)
    return xOffset,yOffset

def createPath(CostSurfacefn,costSurfaceArray,startCoord,stopCoord):

    # coordinates to array index
    startCoordX = startCoord[0]
    startCoordY = startCoord[1]
    startIndexX,startIndexY = coord2pixelOffset(CostSurfacefn,startCoordX,startCoordY)

    stopCoordX = stopCoord[0]
    stopCoordY = stopCoord[1]
    stopIndexX,stopIndexY = coord2pixelOffset(CostSurfacefn,stopCoordX,stopCoordY)

    # create path
    indices, weight = route_through_array(costSurfaceArray, (startIndexY,startIndexX), (stopIndexY,stopIndexX),geometric=True,fully_connected=True)
    indices = np.array(indices).T
    path = np.zeros_like(costSurfaceArray)
    path[indices[0], indices[1]] = 1
    # plt.imshow(path, cmap='hot')
    # plt.show()
    return path

def array2raster(newRasterfn,rasterfn,array):
    raster = gdal.Open(rasterfn)
    geotransform = raster.GetGeoTransform()
    originX = geotransform[0]
    originY = geotransform[3]
    pixelWidth = geotransform[1]
    pixelHeight = geotransform[5]
    cols = array.shape[1]
    rows = array.shape[0]

    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Byte)
    outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(array)
    outRasterSRS = osr.SpatialReference()
    outRasterSRS.ImportFromWkt(raster.GetProjectionRef())
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()

def main_costsurf(CostSurfacefn,outputPathfn,startCoord,stopCoord):

    costSurfaceArray = raster2array(CostSurfacefn) # creates array from cost surface raster
    # plt.imshow(costSurfaceArray, cmap='hot')
    # plt.show()
    pathArray = createPath(CostSurfacefn,costSurfaceArray,startCoord,stopCoord) # creates path array
    array2raster(outputPathfn,CostSurfacefn,pathArray) # converts path array to raster
    return costSurfaceArray

def pixelOffset2coord(rasterfn,xOffset,yOffset):
    raster = gdal.Open(rasterfn)
    geotransform = raster.GetGeoTransform()
    originX = geotransform[0]
    originY = geotransform[3]
    pixelWidth = geotransform[1]
    pixelHeight = geotransform[5]
    coordX = originX+pixelWidth*xOffset
    coordY = originY+pixelHeight*yOffset
    return coordX, coordY

def raster2array(rasterfn):
    raster = gdal.Open(rasterfn)
    band = raster.GetRasterBand(1)
    array = band.ReadAsArray()
    return array

def array2shp(array,outSHPfn,rasterfn,pixelValue):

    # max distance between points
    raster = gdal.Open(rasterfn)
    geotransform = raster.GetGeoTransform()
    pixelWidth = geotransform[1]
    # maxDistance = ceil(sqrt(2*pixelWidth*pixelWidth))
    maxDistance = 0.000999
    # print(maxDistance)

    # array2dict
    count = 0
    roadList = np.where(array == pixelValue)
    multipoint = ogr.Geometry(ogr.wkbMultiLineString)
    pointDict = {}
    for indexY in roadList[0]:
        indexX = roadList[1][count]
        Xcoord, Ycoord = pixelOffset2coord(rasterfn,indexX,indexY)
        pointDict[count] = (Xcoord, Ycoord)
        count += 1

    # dict2wkbMultiLineString
    multiline = ogr.Geometry(ogr.wkbMultiLineString)
    for i in itertools.combinations(pointDict.values(), 2):
        point1 = ogr.Geometry(ogr.wkbPoint)
        point1.AddPoint(i[0][0],i[0][1])
        point2 = ogr.Geometry(ogr.wkbPoint)
        point2.AddPoint(i[1][0],i[1][1])

        distance = point1.Distance(point2)
        if distance < maxDistance:
            line = ogr.Geometry(ogr.wkbLineString)
            line.AddPoint(i[0][0],i[0][1])
            line.AddPoint(i[1][0],i[1][1])
            multiline.AddGeometry(line)

    # wkbMultiLineString2shp
    shpDriver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(outSHPfn):
        shpDriver.DeleteDataSource(outSHPfn)
    outDataSource = shpDriver.CreateDataSource(outSHPfn)
    outLayer = outDataSource.CreateLayer(outSHPfn, geom_type=ogr.wkbMultiLineString )
    featureDefn = outLayer.GetLayerDefn()
    outFeature = ogr.Feature(featureDefn)
    outFeature.SetGeometry(multiline)
    outLayer.CreateFeature(outFeature)


def create_shp(rasterfn,outSHPfn,pixelValue):
    array = raster2array(rasterfn)
    array2shp(array,outSHPfn,rasterfn,pixelValue)


def raster2array_xarray(rasterfn):
    dem = rxr.open_rasterio(rasterfn)
    # dem_array = dem.read(1).astype('float64')
    rds_2 = dem.squeeze().drop("spatial_ref").drop("band")
    

    return rds_2

def _get_pixel_id(point, raster, xdim=None, ydim=None):
    # get location in `raster` pixel space for `point` in y-x coordinate space
    # point: (y, x) - coordinates of the point
    # xdim: name of the x coordinate dimension in input `raster`.
    # ydim: name of the x coordinate dimension in input `raster`

    if ydim is None:
        ydim = raster.dims[-2]
    if xdim is None:
        xdim = raster.dims[-1]
    y_coords = raster.coords[ydim].data
    x_coords = raster.coords[xdim].data

    cellsize_x, cellsize_y = get_dataarray_resolution(raster, xdim, ydim)
    py = int(abs(point[0] - y_coords[0]) / cellsize_y)
    px = int(abs(point[1] - x_coords[0]) / cellsize_x)

    # return index of row and column where the `point` located.
    return py, px

def _is_not_crossable(cell_value, barriers):
    # nan cell is not walkable
    if np.isnan(cell_value):
        return True

    for i in barriers:
        if cell_value == i:
            return True
    return False

def _neighborhood_structure(connectivity=8):
    if connectivity == 8:
        # 8-connectivity
        neighbor_xs = [-1, -1, -1, 0, 0, 1, 1, 1]
        neighbor_ys = [-1, 0, 1, -1, 1, -1, 0, 1]
    else:
        # 4-connectivity
        neighbor_ys = [0, -1, 1, 0]
        neighbor_xs = [-1, 0, 0, 1]
    return np.array(neighbor_ys), np.array(neighbor_xs)


def array2shp_xarray(path_agg,offset,outSHPfn):

    # max distance between points
    # maxDistance = ceil(sqrt(2*pixelWidth*pixelWidth))
    #TODO: this is a specific value that is only suitable for this specific raster 
    # maxDistance = 0.0004
    maxDistance = 1

    # print(maxDistance)

    # count_arr = {0: [0, True], 1: [1, True], 2: [0, False], 3: [1, False]}
    # countList = 0

    # array2dict
    #adds  points to list, gets next point from list 
    count = 0
    roadList = np.where(path_agg >=0)
    multipoint = ogr.Geometry(ogr.wkbMultiLineString)
    # pointDict = {}
    # print([i for i in roadList])
    neighbor_ys,neighbor_xs = _neighborhood_structure(8)

    #from roadlist, start with first x and y
    #get neighbor that is in roadlist
   

    #add neighbor to to line
    #find neighbor from neighbor
    

    pointList=[]
    for indexY in roadList[0]:
        indexX = roadList[1][count]
        Xcoord, Ycoord = path_agg[indexY, indexX].coords['x'].item(), path_agg[indexY, indexX].coords['y'].item()
        # pointDict[count] = (Xcoord, Ycoord)
        if indexY == 60 and indexX == 92:
            print('hi', Xcoord, Ycoord)
        pointList.append((Xcoord, Ycoord))
        count += 1
    # print(pointList)

   


    dtype = [('x', float), ('y', float)]
    values = pointList
    pointArr = np.array(values, dtype=dtype) 
    # dict2wkbMultiLineString
    multiline = ogr.Geometry(ogr.wkbMultiLineString)
    line = ogr.Geometry(ogr.wkbLineString)
    # for i in itertools.combinations(pointDict.values(), 2):
    order='x'
    # for i in itertools.combinations(pointList, 2):
    pl = len(pointList)
    # i = pointList[0]
    coords_neighb = _get_pixel_id(offset, path_agg)
    i=(path_agg[coords_neighb[0], coords_neighb[1]].coords['x'].item(), path_agg[coords_neighb[0], coords_neighb[1]].coords['y'].item())
    # print(coords_neighb, offset)

    # coords_neighb=(roadList[0][0], roadList[1][0])
    roadList_intersect = [(g[0],g[1]) for g in zip(roadList[0], roadList[1])]
    while len(pointList) > 1:
        # print(pl - len(pointList))
        # print(set([(-72.8591666666666, 7.563055555555556)]).intersection(pointList))
        neighbors = [(coords_neighb[0]+a, coords_neighb[1]+b) for a, b in zip(neighbor_ys, neighbor_xs)]
        ngb_set = set(neighbors).intersection(roadList_intersect)
        if not ngb_set:
            # print(neighbors)
            # print(roadList_intersect)
            # print(coords_neighb)
            # print(i)
            # print(i_prev)
            # print(coords_neighb_prev)
            # print(pointList)
            # print(ngb_set)
            break
        
        roadList_intersect.remove(coords_neighb)
        coords_neighb = ngb_set.pop()
        Xcoord, Ycoord = path_agg[coords_neighb[0], coords_neighb[1]].coords['x'].item(), path_agg[coords_neighb[0], coords_neighb[1]].coords['y'].item()

    
        point1 = ogr.Geometry(ogr.wkbPoint)
        # point1.AddPoint(i[0][0],i[0][1])
        point1.AddPoint(i[0],i[1])
        point2 = ogr.Geometry(ogr.wkbPoint)
        # point2.AddPoint(pointList[1][0],pointList[1][1])
        point2.AddPoint(Xcoord, Ycoord)

        distance = point1.Distance(point2)
        # print(point1,point2, distance)
        # print()
        print(distance)

        if distance < maxDistance:
            # print(distance)
        
            # print(point1,point2, distance)
            # line = ogr.Geometry(ogr.wkbLineString)
            line.AddPoint(i[0],i[1])
            # line.AddPoint(pointList[1][0],pointList[1][1])
            # multiline.AddGeometry(line)

            pointList.remove(i)
            i_prev = i
            coords_neighb_prev = coords_neighb
            i = (Xcoord, Ycoord)
        else:
            # print(point1,point2, distance)
            # line = ogr.Geometry(ogr.wkbLineString)
            line.AddPoint(i[0],i[1])
            # line.AddPoint(pointList[1][0],pointList[1][1])
            # multiline.AddGeometry(line)
            # print(i)
            pointList.remove(i)
            i = (Xcoord, Ycoord)

    # print(pointList)
    line.AddPoint(pointList[0][0],pointList[0][1])

            
        # else:
          

        #     pointList = sorted(pointList, key=lambda a: a[count_arr[countList][0]], reverse=count_arr[countList][1])
        #     countList+=1
        #     if countList > 3:
        #         countList = 0  
            

    # wkbMultiLineString2shp
    shpDriver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(outSHPfn):
        shpDriver.DeleteDataSource(outSHPfn)
    outDataSource = shpDriver.CreateDataSource(outSHPfn)
    # print(outDataSource)
    outLayer = outDataSource.CreateLayer(outSHPfn, geom_type=ogr.wkbLineString )
    featureDefn = outLayer.GetLayerDefn()
    outFeature = ogr.Feature(featureDefn)
    outFeature.SetGeometry(line)
    outLayer.CreateFeature(outFeature)



NONE = -1


def _get_pixel_id(point, raster, xdim=None, ydim=None):
    # get location in `raster` pixel space for `point` in y-x coordinate space
    # point: (y, x) - coordinates of the point
    # xdim: name of the x coordinate dimension in input `raster`.
    # ydim: name of the x coordinate dimension in input `raster`

    if ydim is None:
        ydim = raster.dims[-2]
    if xdim is None:
        xdim = raster.dims[-1]
    y_coords = raster.coords[ydim].data
    x_coords = raster.coords[xdim].data

    cellsize_x, cellsize_y = get_dataarray_resolution(raster, xdim, ydim)
    py = int(abs(point[0] - y_coords[0]) / cellsize_y)
    px = int(abs(point[1] - x_coords[0]) / cellsize_x)

    # return index of row and column where the `point` located.
    return py, px


@ngjit
def _is_not_crossable(cell_value, barriers):
    # nan cell is not walkable
    if np.isnan(cell_value):
        return True

    for i in barriers:
        if cell_value == i:
            return True
    return False


@ngjit
def _is_inside(py, px, h, w):
    inside = True
    if px < 0 or px >= w:
        inside = False
    if py < 0 or py >= h:
        inside = False
    return inside


@ngjit
def _distance(x1, y1, x2, y2):
    # euclidean distance in pixel space from (y1, x1) to (y2, x2)
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

@ngjit
def _distance_min_cost(data, x1, y1, x2, y2):
    # euclidean distance in pixel space from (y1, x1) to (y2, x2)
    # print(data[x1,y1], data[x2,y2] )
    # (sqrt(2)/2)*costs[1,1] + (sqrt(2)/2)*costs[2,2]
    return (1/2)*data[y1,x1] + (1/2)*data[y2,x2]

@ngjit
def _distance_min_cost_diag(data, x1, y1, x2, y2):
    # euclidean distance in pixel space from (y1, x1) to (y2, x2)
    # print(data[x1,y1], data[x2,y2] )
    # (sqrt(2)/2)*costs[1,1] + (sqrt(2)/2)*costs[2,2]
    return (np.sqrt(2)/2)*data[y1,x1] + (np.sqrt(2)/2)*data[y2,x2]



@ngjit
def _heuristic(x1, y1, x2, y2):
    # heuristic to estimate distance between 2 point
    # TODO: what if we want to use another distance metric?
    return _distance(x1, y1, x2, y2)

@ngjit
def _heuristic_min_cost(data,x1, y1, x2, y2):
    # heuristic to estimate distance between 2 point
    # TODO: what if we want to use another distance metric?
    return _distance_min_cost(data,x1, y1, x2, y2)


@ngjit
def _min_cost_pixel_id(data, cost, is_open):
    height, width = cost.shape
    py = NONE
    px = NONE
    # set min cost to a very big number
    # this value is only an estimation
    # print(data.max)
    min_cost = (height + width) * data.max().item()
    for i in range(height):
        for j in range(width):
            if is_open[i, j] and cost[i, j] < min_cost:
                min_cost = cost[i, j]
                py = i
                px = j
    return py, px


@ngjit
def _find_nearest_pixel(py, px, data, barriers):
    # if the cell is already valid, return itself
    if not _is_not_crossable(data[py, px], barriers):
        return py, px

    height, width = data.shape
    # init min distance as max possible distance
    min_distance = _distance(0, 0, height - 1, width - 1)
    # return of the function
    nearest_y = NONE
    nearest_x = NONE
    for y in range(height):
        for x in range(width):
            if not _is_not_crossable(data[y, x], barriers):
                d = _distance(x, y, px, py)
                if d < min_distance:
                    min_distance = d
                    nearest_y = y
                    nearest_x = x

    return nearest_y, nearest_x


@ngjit
def _reconstruct_path(path_img, parent_ys, parent_xs, cost,
                      start_py, start_px, goal_py, goal_px):
    # construct path output image as a 2d array with NaNs for non-path pixels,
    # and the value of the path pixels being the current cost up to that point
    current_x = goal_px
    current_y = goal_py

    print(cost)

    if parent_xs[current_y, current_x] != NONE and \
            parent_ys[current_y, current_x] != NONE:
        # exist path from start to goal
        # add cost at start
        path_img[start_py, start_px] = cost[start_py, start_px]
        # add cost along the path
 
        while current_x != start_px or current_y != start_py:
            # value of a path pixel is the cost up to that point
            path_img[current_y, current_x] = cost[current_y, current_x]
            parent_y = parent_ys[current_y, current_x]
            parent_x = parent_xs[current_y, current_x]
            current_y = parent_y
            current_x = parent_x
    return


def _neighborhood_structure(connectivity=8):
    if connectivity == 8:
        # 8-connectivity
        neighbor_xs = [-1, -1, -1, 0, 0, 1, 1, 1]
        neighbor_ys = [-1, 0, 1, -1, 1, -1, 0, 1]
    else:
        # 4-connectivity
        neighbor_ys = [0, -1, 1, 0]
        neighbor_xs = [-1, 0, 0, 1]
    return np.array(neighbor_ys), np.array(neighbor_xs)


def _check_proximity(y,x, agg, goal):
    # print(y,x)

    # print(agg.x.size)
    # print(agg.y.size)
    var=100
    # print(x-var)
    if x-var > 0:
        xvar1 = x-var
    else:
        xvar1 = 0

    if y-var > 0:
        yvar1 = y-var
    else:
        yvar1 = 0

    if x+var < agg.x.size:
        xvar2 = x+var
    else:
        xvar2 = agg.x.size

    if y+var < agg.y.size:
        yvar2 = y+var
    else:
        yvar2 = agg.y.size

    # print(yvar1, yvar2, xvar1, xvar2)

    agg = agg[yvar1:yvar2, xvar1:xvar2]

    # print(agg)
    # print(agg.where(agg.isin(goal), drop=True))

    
    if agg.isin(goal).sum() > 0:
        return True
    else:
        return False




@ngjit
def _a_star_search_py(data, path_img, start_py, start_px, goal,
                   barriers, neighbor_ys, neighbor_xs):

    height, width = data.shape
    # print(data.shape)
    # print(data[start_py, start_px])
    # parent of the (i, j) pixel is the pixel at
    # (parent_ys[i, j], parent_xs[i, j])
    # first initialize parent of all cells as invalid (NONE, NONE)
    parent_ys = np.ones((height, width), dtype=np.int64) * NONE
    parent_xs = np.ones((height, width), dtype=np.int64) * NONE

    # parent of start is itself
    parent_ys[start_py, start_px] = start_py
    parent_xs[start_py, start_px] = start_px

    # distance from start to the current node
    d_from_start = np.zeros_like(data, dtype=np.float64)
    # total cost of the node: cost = d_from_start + d_to_goal
    # heuristic — estimated distance from the current node to the end node
    cost = np.zeros_like(data, dtype=np.float64)

    # initialize both open and closed list all False
    is_open = np.zeros(data.shape, dtype=np.bool_)
    is_closed = np.zeros(data.shape, dtype=np.bool_)




    if not _is_not_crossable(data[start_py, start_px], barriers):
        # print('crossable')
        # if start node is crossable
        # add the start node to open list
        is_open[start_py, start_px] = True
        # init cost at start location
        d_from_start[start_py, start_px] = 0
        cost[start_py, start_px] = d_from_start[start_py, start_px] #+ _heuristic_min_cost(data, start_px, start_py, goal_px, goal_py)
        # print( cost[start_py, start_px])
            # _heuristic(start_px, start_py, goal_px, goal_py) 



    num_open = np.sum(is_open)
    count=0

    while num_open > 0:
        # print(count)
        count +=1
        #TODO: create limit variable
        if count == 40000:
            # print('none found')
            return
        py, px = _min_cost_pixel_id(data, cost, is_open)
        # print(py, px)
        # pop current node off open list, add it to closed list
        is_open[py][px] = 0
        is_closed[py][px] = True
        
        # found the goal
        # print(data[py][px])
        if data[py][px] == goal:
            # reconstruct path
            # print(start_py, start_px, py, px)
            _reconstruct_path(path_img, parent_ys, parent_xs,
                              d_from_start, start_py, start_px,
                              py, px)
            # print('goal found')
            return 
        # if (py, px) == (goal_py, goal_px):
        #     # reconstruct path
        #     _reconstruct_path(path_img, parent_ys, parent_xs,
        #                       d_from_start, start_py, start_px,
        #                       goal_py, goal_px)
        #     return

        # visit neighborhood
        # neighb_dct={}
        for y, x in zip(neighbor_ys, neighbor_xs):
            neighbor_y = py + y
            neighbor_x = px + x
            if neighbor_y > height - 1 or neighbor_y < 0 \
                    or neighbor_x > width - 1 or neighbor_x < 0:
                continue

            # walkable
            if _is_not_crossable(data[neighbor_y][neighbor_x], barriers):
                continue

            # check if neighbor is in the closed list
            if is_closed[neighbor_y, neighbor_x]:
                continue
            # distance from start to this neighbor
            if abs(neighbor_y + neighbor_x) == 1:
                d = d_from_start[py, px] + _distance_min_cost(data, px, py,
                                                 neighbor_x, neighbor_y)
            else:
                d = d_from_start[py, px] + _distance_min_cost_diag(data, px, py,
                                                 neighbor_x, neighbor_y)
            # if neighbor is already in the open list
            # print(f'd: {d}, dfromstrat: {d_from_start[neighbor_y, neighbor_x]}')
            # print('distance1'+d)
            # print(d, d_from_start[neighbor_y, neighbor_x])
            if is_open[neighbor_y, neighbor_x] and \
                    d > d_from_start[neighbor_y, neighbor_x]:
                continue

            


            # neighb_dct[neighbor_y, neighbor_x] = d
        
        # neighb_dct = dict(sorted(neighb_dct.items(), key=lambda item: item[1]))
        # print(neighb_dct)

        # for neighbor_y, neighbor_x in neighb_dct.keys():

            # print(neighbor_x)
            # neighbor is within the surface image

            

            # calculate cost
            # print(is_open[neighbor_y, neighbor_x], is_closed[neighbor_y, neighbor_x], d, d_from_start[neighbor_y, neighbor_x])
            d_from_start[neighbor_y, neighbor_x] = d #neighb_dct[neighbor_y, neighbor_x]
            # d_to_goal = _heuristic_min_cost(data, neighbor_x, neighbor_y, goal_px, goal_py)
            # d_to_goal = _heuristic(neighbor_x, neighbor_y, goal_px, goal_py)
            

            cost[neighbor_y, neighbor_x] = \
                d_from_start[neighbor_y, neighbor_x] #+ d_to_goal
            # add neighbor to the open list
            is_open[neighbor_y, neighbor_x] = True
            parent_ys[neighbor_y, neighbor_x] = py
            parent_xs[neighbor_y, neighbor_x] = px

        num_open = np.sum(is_open)
    # print('none found 2')
    return


def a_star_search_py(surface: xr.DataArray,
                  start: Union[tuple, list, np.array],
                  goal: Union[tuple, list, np.array],
                  barriers: list = [],
                  x: Optional[str] = 'x',
                  y: Optional[str] = 'y',
                  connectivity: int = 8,
                  snap_start: bool = False,
                  snap_goal: bool = False) -> xr.DataArray:
    """
    Calculate distance from a starting point to a goal through a
    surface graph. Starting location and goal location should be within
    the graph.

    A* is a modification of Dijkstra’s Algorithm that is optimized for
    a single destination. Dijkstra’s Algorithm can find paths to all
    locations; A* finds paths to one location, or the closest of several
    locations. It prioritizes paths that seem to be leading closer to
    a goal.

    The output is an equal sized Xarray.DataArray with NaNs for non-path
    pixels, and the value of the path pixels being the current cost up
    to that point.

    Parameters
    ----------
    surface : xr.DataArray
        2D array of values to bin.
    start : array-like object of 2 numeric elements
        (y, x) or (lat, lon) coordinates of the starting point.
    goal : array like object of 2 numeric elements
        (y, x) or (lat, lon) coordinates of the goal location.
    barriers : array like object, default=[]
        List of values inside the surface which are barriers
        (cannot cross).
    x : str, default='x'
        Name of the x coordinate in input surface raster.
    y: str, default='x'
        Name of the y coordinate in input surface raster.
    connectivity : int, default=8
    snap_start: bool, default=False
        Snap the start location to the nearest valid value before
        beginning pathfinding.
    snap_goal: bool, default=False
        Snap the goal location to the nearest valid value before
        beginning pathfinding.

    Returns
    -------
    path_agg: xr.DataArray of the same type as `surface`.
        2D array of pathfinding values.
        All other input attributes are preserved.

    References
    ----------
        - Red Blob Games: https://www.redblobgames.com/pathfinding/a-star/implementation.html  # noqa
        - Nicholas Swift: https://medium.com/@nicholas.w.swift/easy-a-star-pathfinding-7e6689c7f7b2  # noqa

    Examples
    --------
    ... sourcecode:: python

        >>> import numpy as np
        >>> import xarray as xr
        >>> from xrspatial import a_star_search
        >>> agg = xr.DataArray(np.array([
        ...     [0, 1, 0, 0],
        ...     [1, 1, 0, 0],
        ...     [0, 1, 2, 2],
        ...     [1, 0, 2, 0],
        ...     [0, 2, 2, 2]
        ... ]), dims=['lat', 'lon'])
        >>> height, width = agg.shape
        >>> _lon = np.linspace(0, width - 1, width)
        >>> _lat = np.linspace(height - 1, 0, height)
        >>> agg['lon'] = _lon
        >>> agg['lat'] = _lat

        >>> barriers = [0]  # set pixels with value 0 as barriers
        >>> start = (3, 0)
        >>> goal = (0, 1)
        >>> path_agg = a_star_search(agg, start, goal, barriers, 'lon', 'lat')
        >>> print(path_agg)
        <xarray.DataArray (lat: 5, lon: 4)>
        array([[       nan,        nan,        nan,        nan],
               [0.        ,        nan,        nan,        nan],
               [       nan, 1.41421356,        nan,        nan],
               [       nan,        nan, 2.82842712,        nan],
               [       nan, 4.24264069,        nan,        nan]])
        Coordinates:
          * lon      (lon) float64 0.0 1.0 2.0 3.0
          * lat      (lat) float64 4.0 3.0 2.0 1.0 0.0
    """

    if surface.ndim != 2:
        raise ValueError("input `surface` must be 2D")

    if surface.dims != (y, x):
        raise ValueError("`surface.coords` should be named as coordinates:"
                         "({}, {})".format(y, x))

    if connectivity != 4 and connectivity != 8:
        raise ValueError("Use either 4 or 8-connectivity.")

    # convert starting and ending point from geo coords to pixel coords
    start_py, start_px = _get_pixel_id(start, surface, x, y)
    # goal_py, goal_px = _get_pixel_id(goal, surface, x, y)

    h, w = surface.shape
    # validate start and goal locations are in the graph
    if not _is_inside(start_py, start_px, h, w):
        # print(f"start location, {start_px,start_py} outside the surface graph.")
        warnings.warn(f"start location, {start_px,start_py} outside the surface graph.", Warning)
        # raise ValueError("start location outside the surface graph.")
        return xr.DataArray()

    # if not _is_inside(goal_py, goal_px, h, w):
    #     raise ValueError("goal location outside the surface graph.")

    barriers = np.array(barriers)

    if snap_start:
        # find nearest valid pixel to the start location
        start_py, start_px = _find_nearest_pixel(
            start_py, start_px, surface.data, barriers
        )
    if _is_not_crossable(surface.data[start_py, start_px], barriers):
        warnings.warn("Start at a non crossable location", Warning)
        return xr.DataArray()
    
    
    if not _check_proximity(start_py, start_px, surface, goal):
        # print('no road in proximity')
        return xr.DataArray()
    # else:
        # print('road in proximity')
    

    # if snap_goal:
    #     # find nearest valid pixel to the goal location
    #     goal_py, goal_px = _find_nearest_pixel(
    #         goal_py, goal_px, surface.data, barriers
    #     )
    # if _is_not_crossable(surface.data[goal_py, goal_px], barriers):
    #     warnings.warn("End at a non crossable location", Warning)
    #     return xr.DataArray()

    # 2d output image that stores the path
    path_img = np.zeros_like(surface, dtype=np.float64)
    # first, initialize all cells as np.nans
    path_img[:] = np.nan

  


    if start_py != NONE:
        neighbor_ys, neighbor_xs = _neighborhood_structure(connectivity)
        _a_star_search_py(surface.data, path_img, start_py, start_px,
                       goal, barriers, neighbor_ys, neighbor_xs)

    path_agg = xr.DataArray(path_img,
                            coords=surface.coords,
                            dims=surface.dims,
                            attrs=surface.attrs)
    

    return path_agg

def connecting(node_lst, pathname, goal, barriers = [0], xarr = True):
    # nearest_lst = nearest_node_loop(node_lst, G)
    nearest_lst = []
        


    count=0
    CostSurfacefn = pathname
    for i in tqdm(node_lst):
        node_id  = int(abs(i.y+i.x)*1e6)
        nearest_lst.append({"x_start" : i.x, "y_start": i.y, "u": node_id})
        outputPathfn = pathname + f'{node_id}.tif'
        if xarr == False:
            startCoord = (i.x, i.y)
            stopCoord = goal
            # print(stopCoord)
            costSurfaceArray = main_costsurf(CostSurfacefn,outputPathfn,startCoord,stopCoord)

            outSHPfn = f'Path{count}.shp'
            pixelValue = 1
            create_shp(outputPathfn,outSHPfn,pixelValue)
        else:
            
            startCoord = (i.y, i.x)
            stopCoord = goal
            # print(startCoord, stopCoord)
            dem_2=raster2array_xarray(CostSurfacefn)
            dem_2=dem_2[1:-1,1:-1]

            path_agg = a_star_search_py(dem_2, startCoord, stopCoord, barriers, 'x', 'y', connectivity=8)
            print(path_agg.data.size)
            print(np.count_nonzero(np.isnan(path_agg.data)))
        
        #if there is only 1 not nan value, start node is same as end node
        if (path_agg.data.size - np.count_nonzero(np.isnan(path_agg.data)))  == 1:
            # print('point on road')
            j = nearest_lst[-1]
            j["geometry"] = Point(i.x,i.y)
            j["length"] = 0
            j["y_goal"] = i.y
            j['x_goal'] = i.x



        
        #if there is more than 1 not nan value, and not all values are filled, then thisss
        elif np.count_nonzero(np.isnan(path_agg.data)) < path_agg.data.size and path_agg.data.size > 1:                
            # print("outputting")
            outSHPfn = pathname + f'/shps/{node_id}.shp'
            array2shp_xarray(path_agg,startCoord,outSHPfn)

            # print("outputtin 2")
            file = ogr.Open(pathname + f'{node_id}.shp')
            shape = file.GetLayer(0)
            #first feature of the shapefile
            feature = shape.GetFeature(0)
            first = feature.ExportToJson()
            # print(first)
            first=eval(first)
            shp_geom = shp(first['geometry'])
            file=None

            
            line_geo = gpd.GeoSeries(shp_geom, crs='EPSG:4326')
            length = line_geo.to_crs('EPSG:21897').length[0]
            j = nearest_lst[-1]
            j["geometry"] = shp_geom
            j["length"] = length
            j["y_goal"] = Point(shp_geom.coords[0]).y
            j['x_goal'] = Point(shp_geom.coords[0]).x
            count+=1
            # print(nearest_lst)
        

    CostSurfacefn = None

    return nearest_lst


def connecting_mp(i, G, pathname, goal, node_type, barriers = [0], xarr = True):
        #TODO: remove redundant algorithm with xarr false
    # nearest_lst = nearest_node_loop(node_lst, G)
        # nearest_lst = {}

            


        count=0
        CostSurfacefn = pathname
        
        node_id  = int(abs(i.y+i.x)*1e6)
        nearest_lst = {"x_start" : i.x, "y_start": i.y, "u": node_id, "type": node_type}
        outputPathfn = pathname + f'{node_id}.tif'
        if xarr == False:
            startCoord = (i.x, i.y)
            stopCoord = goal
            # print(stopCoord)
            costSurfaceArray = main_costsurf(CostSurfacefn,outputPathfn,startCoord,stopCoord)

            outSHPfn = f'Path{count}.shp'
            pixelValue = 1
            create_shp(outputPathfn,outSHPfn,pixelValue)
        else:
            
            startCoord = (i.y, i.x)
            stopCoord = goal
            # print(startCoord, stopCoord)
            dem_2=raster2array_xarray(CostSurfacefn)
            dem_2=dem_2[1:-1,1:-1]

            path_agg = a_star_search_py(dem_2, startCoord, stopCoord, barriers, 'x', 'y', connectivity=8)
            # print(path_agg.data.size)
            # print(np.count_nonzero(np.isnan(path_agg.data)))
            # print(path_agg)

        #if there is only 1 area filled, start and end node are the same
        if (path_agg.data.size - np.count_nonzero(np.isnan(path_agg.data)))  == 1:
            nearest_lst["geometry"] = Point(i.x,i.y)
            nearest_lst["length"] = 0
            nearest_lst["y_goal"] = i.y
            nearest_lst['x_goal'] = i.x
            nearest_lst['goal_id'] = node_id


        
        #if more than 1 are 
        elif np.count_nonzero(np.isnan(path_agg.data)) < path_agg.data.size and path_agg.data.size > 1:               
             # print("outputting")
            outSHPfn = pathname + f'{node_id}.shp'
            array2shp_xarray(path_agg,startCoord,outSHPfn)

            # print("outputtin 2")
            file = ogr.Open(pathname + f'{node_id}.shp')
            shape = file.GetLayer(0)
            #first feature of the shapefile
            feature = shape.GetFeature(0)
            first = feature.ExportToJson()
            # print(first)
            first=eval(first)
            shp_geom = shp(first['geometry'])

            
            line_geo = gpd.GeoSeries(shp_geom, crs='EPSG:4326')
            length = line_geo.to_crs('EPSG:21897').length[0]
            nearest_lst["geometry"] = shp_geom
            nearest_lst["length"] = length
            nearest_lst["y_goal"] = Point(shp_geom.coords[-1]).y
            nearest_lst['x_goal'] = Point(shp_geom.coords[-1]).x
            nearest_lst['goal_id'] = int(abs(Point(shp_geom.coords[-1]).y+Point(shp_geom.coords[-1]).x)*1e6)
            count+=1
            file=None

        return nearest_lst



def pathfinding(G, pth_out, inp_nodes, slope_pth, goal, node_type, barriers):
    #algorithm that creates a path from a node to a point at a road or river
    #output: coordinates of start node, coordinates of goal, geometry of path
    
    #finds geometry of path to a road or water thingie
    if os.path.exists(pth_out):
            with open(pth_out,'rb') as f:
                node_list = pickle.load(f)
    else:
        
        num_cores = 7
        node_list = []
        print("else")
        # if __name__ == "__main__":
        print("main")
        #TODO: create external limit distance variable
        #TODO: tqdm werkt niet?
        #TODO: G seems redundant
        processed_list = Parallel(n_jobs=num_cores)(delayed(connecting_mp)(inp, G, slope_pth, goal, node_type, barriers) for inp in tqdm(inp_nodes))
        
        for node in processed_list:
            node_list.append(node)
    

        with open(pth_out,'wb') as f:
            pickle.dump(node_list, f)

    return node_list



