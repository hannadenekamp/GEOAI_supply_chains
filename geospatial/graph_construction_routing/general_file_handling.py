from shapely.geometry import box
import rasterio as rio
from tqdm import trange, tqdm
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import split
import rioxarray as rxr
import numpy as np
import osmnx as ox


#reads a dem file
def read_rast_rasterio(filepath): 
    dem = rio.open(filepath)
    dem_array = dem.read(1).astype('float64')
    return dem, dem_array

def read_rast_rxr(filepath):
    return rxr.open_rasterio(filepath)


#creates a polygon of the specified bbox
def create_geom(bounds):
    geom = box(*bounds)
    return geom

#OSMNX module to get a graph from the specified polygon
def get_graph(geom):
    G = ox.graph_from_polygon(geom, retain_all=True, truncate_by_edge=True)
    return G


def add_nodes(nodes_lst, G):
    G.add_nodes_from(nodes_lst)
    return G

#simplification of the graph to consolidate the many nodes( especially inside cities) in a range of 100 meters
def consol_graph(G):
    G_proj = ox.project_graph(G)
    G_proj = ox.simplification.consolidate_intersections(G_proj,tolerance=100, dead_ends=True)
    # ox.plot_graph(G_proj)
    return G_proj

#adds elevation to nodes based on the the DEM
def add_elev(G_proj, pathname):
    G_proj = ox.project_graph(G_proj, to_crs=4326)
    # (print(G_proj.nodes))
    G_proj = ox.add_node_elevations_raster(G_proj,pathname+"_dem.tif" )
    return G_proj


def plot_consol_graph(G_proj, dem, dem_array, ns):
    # Plot uncropped array
    f, ax = plt.subplots(figsize=(20,20))
    
    ep.plot_bands(dem_array,
            cmap='Greys_r',
            ax=ax,
            extent=plotting_extent(dem))  # Use plotting extent from DatasetReader object

    nc = ox.plot.get_node_colors_by_attr(G_proj, 'elevation', cmap='plasma')
    ox.plot_graph(G_proj, figsize=(16, 16), node_color=nc, node_size=ns, node_zorder=2, edge_color='#dddddd', ax=ax)
    plt.show()




def clip_raster(pathname, geom_small):
    # Read raster using rioxarray
    raster = read_rast_rxr(pathname)
    # # Shapely Polygon  to clip raster
    # geom = Polygon([[-13315253,3920415], [-13315821.7,4169010.0], [-13019053.84,4168177.65], [-13020302.1595,3921355.7391]])
    
    # Use shapely polygon in clip method of rioxarray object to clip raster
    clipped_raster = raster.rio.clip([geom_small])
    
    # Save clipped raster
    path_to_tif_file = pathname + "_clipped.tif"

    # # Write the data to a new geotiff file
    # with open(path_to_tif_file, 'w') as outfile:
    clipped_raster.rio.to_raster(path_to_tif_file)



def splitPolygon(polygon, nx, ny):
    minx, miny, maxx, maxy = polygon.bounds
    dx = (maxx - minx) / nx
    dy = (maxy - miny) / ny

    minx, miny, maxx, maxy = polygon.bounds
    dx = (maxx - minx) / nx  # width of a small part
    dy = (maxy - miny) / ny  # height of a small part
    horizontal_splitters = [LineString([(minx, miny + i*dy), (maxx, miny + i*dy)]) for i in range(ny)]
    vertical_splitters = [LineString([(minx + i*dx, miny), (minx + i*dx, maxy)]) for i in range(nx)]
    splitters = horizontal_splitters + vertical_splitters
    result = polygon
    
    for splitter in splitters:
        result = MultiPolygon(split(result, splitter))
    
    return result

def clipping_bbox(bbox, nr_tiles):
    # bbox = [-76.026, 7.711, -65.391, 12.533]

    xmin, ymin, xmax, ymax = bbox
    bbox_lst = []
    width = (xmax-xmin)
    height = (ymax-ymin)
    
    xmin_init = xmin
    # xincr, yincr = /nr_tiles , (ymax-ymin)/nr_tiles
    #times 1.1 so it is 10% larger 
    incr =  np.sqrt(width * height/nr_tiles)*1.1
    # w_h = (xmax-xmin)/(ymax-ymin)
    print(width, height, incr)

    for i in range(np.ceil(height/incr).astype(int)):
        ymax=ymin+incr

        for i in range(np.ceil(width/incr).astype(int)):
            xmax = xmin+incr
            bbox = [xmin, ymin,xmax , ymax]
            xmin = xmin+incr
            bbox_lst.append(bbox)
        xmin = xmin_init
        ymin = ymin+incr

    return bbox_lst

