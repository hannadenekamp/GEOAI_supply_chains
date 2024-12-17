import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

from xrspatial import slope, hillshade
import xarray as xr

import datashader as ds

from datashader.transfer_functions import shade
from datashader.transfer_functions import stack
from datashader.transfer_functions import dynspread
from datashader.transfer_functions import set_background
from datashader.colors import Elevation
import xrspatial


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

from rasterio.features import rasterize
from rasterio.transform import from_bounds
import geopandas as gpd

import rasterio as rio

import connecting_nodes as cn
import general_file_handling as gfh

# def create_slope(pathname, out_pth, slope_type):
#     #TODO: assign geotransform
#     dataset = rio.open(pathname+"_dem.tif")
#     data = dataset.read()
#     data = np.squeeze(data)
#     sp_dem = rd.rdarray(data, no_data=0)
#     slope = rd.TerrainAttribute(sp_dem, attrib=slope_type, zscale = 111120 )
#     profile = dataset.profile
#     profile['nodata'] = -9999.
#     profile['dtype'] = slope.dtype
#     profile
#     profile['transform'] = dataset.transform
#     with rio.open(out_pth, 'w', **profile) as dst:
#         dst.write(slope, 1)


def create_slope(pathname, out_pth, slope_type):
    #TODO: assign geotransform
    rast = gfh.read_rast_rxr(pathname)
    rast.attrs["res"]=30
    rast = rast.squeeze()
    risky = slope(rast)

    dataset = rio.open(pathname)
    profile = dataset.profile
    profile['nodata'] = -9999.
    profile['dtype'] = risky.dtype
    profile
    profile['transform'] = dataset.transform
    
    with rio.open(out_pth, 'w', **profile) as dst:
        dst.write(risky, 1)



def mask_xr(mask_arr_pth, arr_pth, mask_in_val, mask_out_val, align=True):
    arr =cn.raster2array_xarray(arr_pth)
    #TODO: specify align function what it does, maybe rename
    if align == True:
        mask, mask_arr = gfh.read_rast_rasterio(mask_arr_pth)
        mask_arr = xr.DataArray(mask_arr, dims=('y', 'x'), coords={'x': arr.coords['x'].values, 'y': arr.coords['y'].values})
        #creates an array with 0 or 1 where given input value is 
        agg_mask_bool = mask_arr.isin(mask_in_val)

    else:
        # mask_arr =cn.raster2array_xarray(mask_arr_pth)
        mask, mask_arr = gfh.read_rast_rasterio(mask_arr_pth)
        mask_arr = xr.DataArray(mask_arr, dims=('y', 'x'), coords={'x': arr.coords['x'].values, 'y': arr.coords['y'].values})

        agg_mask_bool = mask_arr > (mask_in_val)

    #sets value where in mask array is 1 to the mask output value
    agg_2 = xr.where(agg_mask_bool, mask_out_val, arr)
    return agg_2


def gdf_to_rast(shape, gdf, outpath):
    # Load some sample data
    # lose a bit of resolution, but this is a fairly large file, and this is only an example.
    # shape = (dem_area.height, dem_area.width)
    # shape= 700, 700
    print(gdf['geometry'].total_bounds)
    # transform = rio.transform.from_bounds(*gdf['geometry'].total_bounds, *shape)
    #TODO: make these values external input
    transform= rio.transform.from_bounds(-74.0001388888889, 7.999861111111111, -72.99986111111112, 9.00013888888889, *shape)
    rasterize_gdf= rasterize(
        [(geo_shp, 1) for geo_shp in gdf['geometry']],
        out_shape=shape,
        transform=transform,
        fill=0,
        all_touched=False,
        dtype=rio.uint8)

    with rio.open(
        outpath, 'w',
        driver='GTiff',
        dtype=rio.float32,
        count=1,
        width=shape[0],
        height=shape[1],
        transform=transform
    ) as dst:
        dst.write(rasterize_gdf, indexes=1)



