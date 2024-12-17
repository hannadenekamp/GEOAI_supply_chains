import networkx as nx
import osmnx as ox
from shapely.geometry import Point, LineString
import itertools
import geopandas as gpd



def get_edges_wth_attr(G, attr_name, attr_val):
    attr = nx.get_edge_attributes(G, attr_name)
    edge_lst=[k for k,v in attr.items() if v == attr_val]

                
    return edge_lst

def get_nodes_wth_attr(G, attr_name, attr_val):
    attr = nx.get_node_attributes(G, attr_name)
    node_lst=[k for k,v in attr.items() if v == attr_val]

                
    return node_lst



def assign_edge_attr(G, attrs_dct):
    for k in attrs_dct.keys():
        attr_lst = nx.get_edge_attributes(G, k)

        # for v in attrs_dct[k]: 
        #     edge_lst  = get_edges_wth_attr (G,k, v)
        #TODO:create df -  there calculate the values, then change back into graph?

def connect_edges(G,L, df, graph_counter_l, pathname, edgetype):
    
    G=G.copy()
    if edgetype == 'transhipment_airstrip':
        df.reset_index(inplace=True)

        df_clipped = df.loc[df.osmid.isin(L)]
    else:
        df_clipped = df.copy()
    for i, row in df_clipped.iterrows():
        if edgetype == 'transhipment_airstrip':
            x = row['geometry'].coords[0][0]
            y= row['geometry'].coords[0][1]
            u = row['osmid']
        else:
            x = row['LocationLongitude']
            y= row['LocationLatitude']
            u = row['u']

        G.add_node(row['connect_id'], 
                   x = x, 
                   y= y,
                   type=edgetype)
        G.add_edge(u, row['connect_id'],
                   osmid = [i, row['connect_id']],
                   length = 0,
                   geometry = LineString([Point(x,y), Point(x,y)]),
                   type=edgetype
                   )
    ox.io.save_graphml(G, filepath= pathname+f'_graph_{graph_counter_l}_l.osm', encoding='utf-8')
    nodesAt5 = [x for x,y in G.nodes(data=True) if y['type']== 'airstrip']

    graph_counter_l+=1

    return graph_counter_l

# def connect_edges_ports(G,L, df):
    
#     G=G.copy()
#     for i, row in df:
#         x = row['LocationLatitude']
#         y= row['LocationLongitude']
#         G.add_node(row['connect_id'], 
#                    x = x, 
#                    y= y,
#                    type='port')
#         G.add_edge(i, row['connect_id'],
#                    osmid = [i, row['connect_id']],
#                    length = 0,
#                    geometry = LineString(Point(x,y), Point(x,y)),
#                    type = "transhipment_port"
#                    )

#     return G


def complete_graph_from_list(G, L, df, pathname,graph_counter_l, create_using=None):
    df_clipped = df.loc[df.osmid.isin(L)]
    L_connecting = df_clipped['osmid'].to_list()

    G= G.copy()
    if len(L)>1:
        if G.is_directed():
            edges = itertools.permutations(L_connecting,2)
        else:
            edges = itertools.combinations(L_connecting,2)
    
    for edge in edges:
        u,v = edge

        u_coords = df_clipped.loc[df_clipped.osmid == u].geometry_points.values[0]
        v_coords = df_clipped.loc[df_clipped.osmid == v].geometry_points.values[0]

        x1,y1= u_coords.x, u_coords.y
        x2, y2 =  v_coords.x, v_coords.y
        geom = LineString([Point(x1,y1),Point(x2,y2)])
        gdf = gpd.GeoSeries(geom, crs='EPSG:4326')
        distance = gdf.to_crs('EPSG:21897').length[0]
        G.add_edge(u,v, osmid = [u,v], length =  distance, geometry=geom, type= 'air')

    ox.io.save_graphml(G, filepath= pathname+f'_graph_{graph_counter_l}_l.osm', encoding='utf-8')
    graph_counter_l+=1

        # G.add_edges_from(edges)
    return graph_counter_l



"""
    attrs = {
    node0: {attr0: val00, attr1: val01},
    node1: {attr0: val10, attr1: val11},
    node2: {attr0: val20, attr1: val21},
}
nx.set_node_attributes(G, attrs_dct):
    #TODO: attrs_dct - column names: [potential values]
    

    df_attributes_only = pd.DataFrame(
    [['jim', 'tall', 'red', 'fat'], ['john', 'small', 'blue', 'fat']],
    columns=['id', 'attribute1', 'attribute2', 'attribute3']
)
node_attr = df_attributes_only.set_index('id').to_dict('index')
nx.set_node_attributes(g, node_attr)

g.nodes['jim']
"""

def get_graph_with_attr(G,mode,attr_name_dct):
    ne_lst = []
    if mode == 'node':

        for attr_name in attr_name_dct.keys():
            for attr_val in attr_name_dct[attr_name]:
                nodes = get_nodes_wth_attr(G,attr_name, attr_val)
                ne_lst.extend(nodes)
        G_clipped = G.subgraph(ne_lst).copy()
    if mode == 'edge':
        for attr_name in attr_name_dct.keys():
            for attr_val in attr_name_dct[attr_name]:

                edges = get_edges_wth_attr(G,attr_name, attr_val)
                ne_lst.extend(edges)

        G_clipped = G.edge_subgraph(ne_lst).copy()


    return G_clipped