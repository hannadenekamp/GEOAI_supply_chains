from collections import Counter
import networkx as nx
import general_file_handling as gfh
from shapely.geometry import Polygon
import matplotlib.pyplot as plt


def validate_node_attr(G, attr):
    node_types = nx.get_node_attributes(G, attr)

    attr_count = Counter(node_types.values())

    return attr_count

def validate_edge_attr(G, attr):
    edge_types = nx.get_edge_attributes(G, attr)

    attr_count = Counter(edge_types.values())

    return attr_count

def val_bbox(bbox, nr_tiles):
    bbox_lst = gfh.clipping_bbox(bbox, nr_tiles)
    # bbox = [[float(0), float(3), float(9), float(4)]]
    pol_lst = [Polygon([[long0, lat0],
                            [long1,lat0],
                            [long1,lat1],
                            [long0, lat1]]) for long0, lat0, long1, lat1 in bbox_lst]
    
    print(f'Number of bboxes: {len(bbox_lst)}')

    fig, ax = plt.subplots()
    for pol in pol_lst[:]:
        ax.plot(*pol.exterior.xy)
        ax.set_aspect('equal', adjustable='box')
    return fig