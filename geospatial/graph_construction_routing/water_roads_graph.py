import shapely as shp
from shapely import contains
from shapely.geometry import Point, LineString
from shapely.ops import nearest_points
import osmnx as ox
from tqdm.notebook import trange, tqdm
import geopandas as gpd
import pandas as pd
import traceback
import os
import networkx as nx
from collections import Counter
from collections import defaultdict
import pickle





def get_nodes_near_water(lines_gdf, points_gdf):
    #TODO: change crs??
    lines_buffered = lines_gdf.geometry.buffer(0.002)
    all_nodes = set()
    # points_gdf = points_gdf.reset_index()

    for line in tqdm(lines_buffered):
        
        node_list = [i for i in points_gdf.index if contains(line, points_gdf.loc[i].geometry)]
        all_nodes.update(node_list)

        # all_nodes.append(node_list)
    return lines_buffered, all_nodes

def get_nodes_near_water_mp(i, lines_gdf, points_gdf, buffer_val):
    lines_buffered = lines_gdf.loc[i].geometry.buffer(buffer_val)
    # points_gdf = points_gdf.reset_index()

    
   
    node_list = [j for j in points_gdf.index if contains(lines_buffered, points_gdf.loc[j].geometry)]
    # distance_list = [j for j in points_gdf.index if contains(lines_buffered, points_gdf.loc[j].geometry)]

    return node_list

"""
#G_split: the graph with the edges that need to be split/intersection created
def create_crossing(G_split, X, Y, osmid_in=None, mode=None, G_append = None, edge_attr=None, new_strt_node=None, distance_limit = None):
    G=G_split.copy()
    # print(G.nodes)
    #TODO: project the things first
    edge, distance = ox.distance.nearest_edges(G, X, Y, return_dist = True)
    
    # G[u][v]['color']
    #get nearest edge
    #then get nearest point on edge
    # then use algorithms below to split edge
    print(f'edge: {edge}')
    u,v,i = edge


    try:
        print("trying")

        edge_geom = G[u][v][i]['geometry']
        edge_attrs = G[u][v][i] # Copy edge attributes
        # print(G.nodes[u], G.nodes[v])

        # coords = [Point(x,y)for x,y in edge.coords]

        # The points are returned in the same order as the input geometries:
        # dont know if correct or if needs to be projected first??

        p1, p2 = nearest_points(edge_geom, Point(X,Y))
        # x, y = [x for x in p1.coords]
        osmid =G[u][v][i]['osmid']
        node_type = G[u][v][i]['type']

        
        point_list = [p1]

        


        segments, bool =split_line_with_points(edge_geom, point_list)
        count=1
        print(edge_attrs)
        if (mode == "airstrip") or (mode == "water"):

            #TODO: check if X,Y is correct
            line_geo = gpd.GeoSeries(LineString([p1, Point(X,Y)]), crs='EPSG:4326')
            distance = line_geo.to_crs('EPSG:21897').length[0]

            if distance > distance_limit:
                print(f'too far away {distance}')
                return [osmid_in]



        if( mode == 'new node') or (mode == 'airstrip'):
            print('new start node added')
            if new_strt_node['goal_id'] == new_strt_node['u']:
                osmid_in = osmid_in+new_strt_node['goal_id']

            G_append.add_node(osmid_in, y= new_strt_node['y_start'], x= new_strt_node['x_start'], type=new_strt_node['type'])
            G.add_node(osmid_in, y= new_strt_node['y_start'], x= new_strt_node['x_start'], type=new_strt_node['type'])

        
        #bool that states whether line can be split 
        if bool == True:
            
            #TODO: check if this is correct for all modes
            new_osmid = int(abs(p1.y+p1.x)*1e6)

            
            G_append.add_node(new_osmid, y= p1.y, x= p1.x, type=node_type)
            G.add_node(new_osmid, y= p1.y, x= p1.x, type=node_type)

            for i in segments:
                #TODO: check if with adding edge attributes this still works 
                line_geo = gpd.GeoSeries(i, crs='EPSG:4326')

                length = line_geo.to_crs('EPSG:21897').length[0]
                if count==1:
                    G_append.add_edge(u,new_osmid, **{**edge_attrs, 'length': length, 'geometry': i} )
                    G.add_edge(u,new_osmid, **{**edge_attrs, 'length': length, 'geometry': i} )
                else:
                    G_append.add_edge(new_osmid,v, **{**edge_attrs, 'length': length, 'geometry': i} )
                    G.add_edge(new_osmid,v, **{**edge_attrs, 'length': length, 'geometry': i} )
                count+=1

            #if water==True than there is an edge added between a road node and a (new) water node
            #TODO: create modes, also one for adding new point and road
            if (mode == "water") or (mode == "airstrip" ):
                print(f'mode is water, edge should be added in theory, edge attr: {edge_attr}')
                G_append.add_edge(osmid_in, new_osmid, osmid = [osmid_in, new_osmid], length = distance, geometry = LineString([Point(X,Y),p1]), type = edge_attr)
                G.add_edge(osmid_in, new_osmid, osmid = [osmid_in, new_osmid], length = distance, geometry = LineString([Point(X,Y),p1]), type = edge_attr)
            
            if mode == "new node":
                
                G_append.add_edge(osmid_in, new_osmid, osmid = [osmid_in, new_osmid], length =new_strt_node['length'], geometry = new_strt_node['geometry'], type = edge_attr)
                G.add_edge(osmid_in, new_osmid, osmid = [osmid_in, new_osmid], length =new_strt_node['length'], geometry = new_strt_node['geometry'], type = edge_attr)


            
            print(u,v)
            G.remove_edge(*edge)
            print(f'G normal removed edge {u}, {v}')

            G_append.remove_edge(*edge)
            print(f'G_apppend remove edge {u}, {v}')


            return  [G, p1, Point(X,Y), distance, new_osmid, G_append]
       
        #TODO: bool does not mean that there is no connection, but it means that line is not split but connection is to end node
        # or it means that connection is already there ??
        else:
            #TODO: causes key error when combining water and roads

            geometries = nx.get_node_attributes(G_append, "y")
            print(f'geometries: {u}, {v}, {geometries}')

            if geometries[u] ==  p1.y:
                p_goal = edge_geom.coords[0]
                new_osmid = u

            elif geometries[v] ==  p1.y:
                p_goal = edge_geom.coords[-1]
                new_osmid = v

            else:
                print('no match')
                return [osmid_in]




            #if water==True than there is an edge added between a road node and a (new) water node
            #TODO: create modes, also one for adding new point and road


            if (mode == "water") or (mode == "airstrip" ):
                G_append.add_edge(osmid_in, new_osmid, osmid = [osmid_in, new_osmid], length = distance, geometry = LineString([Point(X,Y),p_goal]), type = edge_attr)
                G.add_edge(osmid_in, new_osmid, osmid = [osmid_in, new_osmid], length = distance, geometry = LineString([Point(X,Y),p_goal]), type = edge_attr)
            elif (mode == "new_node"):
                G_append.add_edge(osmid_in, new_osmid, osmid = [osmid_in, new_osmid], length =new_strt_node['length'], geometry = new_strt_node['geometry'], type = edge_attr)
                G.add_edge(osmid_in, new_osmid, osmid = [osmid_in, new_osmid], length =new_strt_node['length'], geometry = new_strt_node['geometry'], type = edge_attr)
            else:
                print('still no match')


            return  [G, p1, Point(X,Y), distance, new_osmid, G_append]
    except Exception:
        traceback.print_exc()             
        return [osmid_in]
"""



def cut(line, distance):
    # Cuts a line in two at a distance from its starting point
    # This is taken from shapely manual
    # print(distance)
    if distance <= 0.0 or distance >= line.length:
        return [[LineString(line)],False]
    coords = list(line.coords)
    for i, p in enumerate(coords):
        pd = line.line_locate_point(Point(p)) #used to be line.project
        if pd == distance:
            return [[LineString(coords[:i+1]),LineString(coords[i:])], True]
        if pd > distance:
            cp = line.interpolate(distance)
            return [[LineString(coords[:i] + [(cp.x, cp.y)]),LineString([(cp.x, cp.y)] + coords[i:])], True]

def split_line_with_points(line, points):
    """Splits a line string in several segments considering a list of points.

    The points used to cut the line are assumed to be in the line string 
    and given in the order of appearance they have in the line string.

    >>> line = LineString( [(1,2), (8,7), (4,5), (2,4), (4,7), (8,5), (9,18), 
    ...        (1,2),(12,7),(4,5),(6,5),(4,9)] )
    >>> points = [Point(2,4), Point(9,18), Point(6,5)]
    >>> [str(s) for s in split_line_with_points(line, points)]
    ['LINESTRING (1 2, 8 7, 4 5, 2 4)', 'LINESTRING (2 4, 4 7, 8 5, 9 18)', 'LINESTRING (9 18, 1 2, 12 7, 4 5, 6 5)', 'LINESTRING (6 5, 4 9)']

    """
    segments = []
    current_line = line
    for p in points:
        d = current_line.line_locate_point(p) #used to be line.project
        seg_list, bool= cut(current_line, d)
        if len(seg_list) > 1:
            seg,current_line = seg_list
            segments.append(seg)
            segments.append(current_line)
        else:
            # print('no cutting')
            
            segments.append(seg_list[0])


    return segments, bool


def get_potential_overside(G_overside,pm,ps,d, osmid_split, mode, G_append, edge_attr):
    #TODO: check projection
    xm,ym = pm.x, pm.y
    # xs,ys = ps.x, ps.y

    X = xm #-xs + xm 
    Y = ym #-ys + ym 

    G=G_overside.copy()

    edge, distance = ox.distance.nearest_edges(G, X, Y, return_dist = True)
    print(edge)

    if distance < d:
        # print('add_crossing')
        
        
        crossing_lst= create_crossing(G, X, Y,osmid_split, mode, G_append, edge_attr)
        #road node from overside needs to be added?
        #TODO: might be redundant now G_append exists
        if len (crossing_lst)>1:
            print('crossing list long enough')
            G_new, pm_new,ps_new, new_osmid, d, G_append =crossing_lst
            # G_new.add_node(osm_new, y= pm_new.y, x= pm_new.x)
            # G_new.add_edge(osmid_split, osm_new, osmid = [osmid_split, osm_new], length = distance, geometry = LineString([pm, pm_new]), type='wr_connect')
            # print('road overside added')        
        else:
            G_new = G
    else:
        G_new = G

    return G_new,G_append


def add_new_roads(lst, G):
    for i in lst:
        G.add_node(i['u'], y=i['y_start'], x=i['x_start'])
        if len(i) > 3:
            G.add_edge(i['u'], i['goal_id'], osmid=[i['u'], i['goal_id']],length=i['length'], geometry=i['geometry'])


    # df_nodes, df_edges = ox.utils_graph.graph_to_gdfs(G, nodes = True)
    # df_new = pd.DataFrame(lst)
    # df_new = df_new.loc[df_new.geometry.notna()]
    
    # df_start = df_new.copy()[["u", "x_start", "y_start"]]
    # df_start = df_start.rename({"u":"osmid", "x_start": "x", "y_start": "y"}, axis=1)
    # df_start['geometry'] = list(zip(df_start.x, df_start.y))
    # df_start['geometry']=df_start['geometry'].apply(lambda x: Point(x))
    # df_start['lat'] = df_start["y"]
    # df_start['lon'] = df_start["x"]
    # df_start.set_index("osmid", inplace=True)

    # df_nodes_add = pd.concat([df_nodes,df_start]) 

    # df_new["key"] = 0
    # df_new = df_new.set_index(["u","v", "key"])

    # df_final = df_new.copy()[["geometry","length"]]
    # df_edges_add = pd.concat([df_edges, df_final])
    return G

#G_split: the graph with the edges that need to be split/intersection created
def create_crossing(G_split, X, Y, osmid_in=None, mode=None, G_append = None, edge_attr=None, new_strt_node=None, distance_limit = None, constructing_phase=None):
    G=G_split.copy()
    # print(G.nodes)
    #TODO: project the things first
    edge, distance = ox.distance.nearest_edges(G, X, Y, return_dist = True)
    
    # G[u][v]['color']
    #get nearest edge
    #then get nearest point on edge
    # then use algorithms below to split edge
    # print(f'edge: {edge}')
    u,v,i = edge


    try:
        # print("trying")

        edge_geom = G[u][v][i]['geometry']
        edge_attrs = G[u][v][i] # Copy edge attributes
        # print(G.nodes[u], G.nodes[v])
        # print(edge_attrs)

        # coords = [Point(x,y)for x,y in edge.coords]

        # The points are returned in the same order as the input geometries:
        # dont know if correct or if needs to be projected first??

        p1, p2 = nearest_points(edge_geom, Point(X,Y))
        # x, y = [x for x in p1.coords]
        osmid =G[u][v][i]['osmid']
        node_type = G[u][v][i]['type']

        
        point_list = [p1]

        


        segments, bool = split_line_with_points(edge_geom, point_list)
        count=1
        # print(edge_attrs)
        if (mode == "airstrip") or (mode == "water") or (mode == "port" ):

            #TODO: check if X,Y is correct
            line_geo = gpd.GeoSeries(LineString([p1, Point(X,Y)]), crs='EPSG:4326')
            distance = line_geo.to_crs('EPSG:21897').length[0]

            if not(mode == 'port')  and (distance > distance_limit):
                # print(f'too far away {distance}')
                return [osmid_in]

        nd_lst = list(G.nodes)


        if( mode == 'new node') or (mode == 'airstrip') or (mode == "port" ):
            # print('new start node added')
            if new_strt_node['goal_id'] == new_strt_node['u']:
                osmid_in = osmid_in+new_strt_node['goal_id']

            if osmid_in in nd_lst:
                osmid_in = osmid_in*2
                if osmid_in in nd_lst:
                    raise Exception('osmid in list')
            
            nd_lst.append(osmid_in)


            G_append.add_node(osmid_in, y= new_strt_node['y_start'], x= new_strt_node['x_start'], type=new_strt_node['type'], constructing_phase= constructing_phase)
            G.add_node(osmid_in, y= new_strt_node['y_start'], x= new_strt_node['x_start'], type=new_strt_node['type'], constructing_phase=constructing_phase)

        
        #bool that states whether line can be split 
        if bool == True:
            
            #TODO: check if this is correct for all modes

            nd_lst = list(G.nodes)
            # print(nd_lst[:5])
            new_osmid = int(abs(p1.y+p1.x)*1e6)

            if new_osmid in nd_lst:
                new_osmid = new_osmid+new_osmid
                if new_osmid in nd_lst:
                    raise Exception('new osmid in list')


            # print(node_type)
            G_append.add_node(new_osmid, y= p1.y, x= p1.x, type=node_type, constructing_phase=constructing_phase)
            G.add_node(new_osmid, y= p1.y, x= p1.x, type=node_type, constructing_phase=constructing_phase)

            for i in segments:
                #TODO: check if with adding edge attributes this still works 
                line_geo = gpd.GeoSeries(i, crs='EPSG:4326')

                length = line_geo.to_crs('EPSG:21897').length[0]
                if count==1:
                    G_append.add_edge(u,new_osmid, **{**edge_attrs, 'length': length, 'geometry': i, 'constructing_phase':constructing_phase} )
                    G.add_edge(u,new_osmid, **{**edge_attrs, 'length': length, 'geometry': i,'constructing_phase':constructing_phase} )
                else:
                    G_append.add_edge(new_osmid,v, **{**edge_attrs, 'length': length, 'geometry': i,'constructing_phase':constructing_phase} )
                    G.add_edge(new_osmid,v, **{**edge_attrs, 'length': length, 'geometry': i,'constructing_phase':constructing_phase} )
                count+=1

            #if water==True than there is an edge added between a road node and a (new) water node
            #TODO: create modes, also one for adding new point and road
            if (mode == "water") or (mode == "airstrip" )or (mode == "port" ):
                # print(f'mode is water, edge should be added in theory, edge attr: {edge_attr}')
                G_append.add_edge(osmid_in, new_osmid, osmid = [osmid_in, new_osmid], length = distance, geometry = LineString([Point(X,Y),p1]), type = edge_attr, constructing_phase=constructing_phase)
                # G.add_edge(osmid_in, new_osmid, osmid = [osmid_in, new_osmid], length = distance, geometry = LineString([Point(X,Y),p1]), type = edge_attr)
            
            if mode == "new node":
                
                G_append.add_edge(osmid_in, new_osmid, osmid = [osmid_in, new_osmid], length =new_strt_node['length'], geometry = new_strt_node['geometry'], type = edge_attr,constructing_phase=constructing_phase)
                # G.add_edge(osmid_in, new_osmid, osmid = [osmid_in, new_osmid], length =new_strt_node['length'], geometry = new_strt_node['geometry'], type = edge_attr)


            
            # print(u,v)
            G.remove_edge(*edge)
            # print(f'G normal removed edge {u}, {v}')

            G_append.remove_edge(*edge)
            # print(f'G_apppend remove edge {u}, {v}')

            # print(attr_count)
            # print([x for x,y in G_append.nodes(data=True) if y['type']=='water_road']



            return  [G, p1, Point(X,Y), distance, new_osmid, G_append]
       
        #TODO: bool does not mean that there is no connection, but it means that line is not split but connection is to end node
        # or it means that connection is already there ??
        else:
            #TODO: causes key error when combining water and roads

            geometries = nx.get_node_attributes(G_append, "y")
            # print(f'geometries: {u}, {v}, {geometries}')

            if geometries[u] ==  p1.y:
                p_goal = edge_geom.coords[0]
                new_osmid = u

            elif geometries[v] ==  p1.y:
                p_goal = edge_geom.coords[-1]
                new_osmid = v

            else:
                # print('no match')
                return [osmid_in]




            #if water==True than there is an edge added between a road node and a (new) water node
            #TODO: create modes, also one for adding new point and road


            if (mode == "water") or (mode == "airstrip" ) or (mode == "port" ):
                G_append.add_edge(osmid_in, new_osmid, osmid = [osmid_in, new_osmid], length = distance, geometry = LineString([Point(X,Y),p_goal]), type = edge_attr)
                G.add_edge(osmid_in, new_osmid, osmid = [osmid_in, new_osmid], length = distance, geometry = LineString([Point(X,Y),p_goal]), type = edge_attr)
            elif (mode == "new node"):
                G_append.add_edge(osmid_in, new_osmid, osmid = [osmid_in, new_osmid], length =new_strt_node['length'], geometry = new_strt_node['geometry'], type = edge_attr)
                G.add_edge(osmid_in, new_osmid, osmid = [osmid_in, new_osmid], length =new_strt_node['length'], geometry = new_strt_node['geometry'], type = edge_attr)
            else:
                print(f'still no match mode: {mode}')


            return  [G, p1, Point(X,Y), distance, new_osmid, G_append]
    except Exception:
        traceback.print_exc()             
        return [osmid_in]

#Connects airport nodes to existing roads
def connecting(G, G_append, pth_out_small, pth_out_large, inp_pths, edge_attr, mode, constructing_phase, distance_limit = None, waterpth = None):
    if os.path.exists(pth_out_small):
        G_new = ox.io.load_graphml(filepath=pth_out_small)
        G_append = ox.io.load_graphml(filepath=pth_out_large)
        if mode == "water":
            with open(waterpth+'nodes_added.pk','rb') as f:
                nodes_added = pickle.load(f)
            with open(waterpth+'oversides.pk','rb') as f:
                edge_lst = pickle.load(f)

        print('file loaded')


    else:
        #TODO: if it takes too long, update it with the splitting polygon algorithm

        not_connected = []
        edge_lst = []
        nodes_added = []
        G_new = G.copy()
        for node in tqdm(inp_pths):
            # print(node)
            # node_attr = nodes_g2.loc[node]
            #len node is to make sure it contains geometry, but it being an airstrip is an exception
            if (len(node) > 4) or (mode == 'airstrip'):
    
                #TODO: update inpute, here and other functions mode
                crossing = create_crossing(G_new, node['x_goal'], node['y_goal'], node['u'], mode, G_append, edge_attr, new_strt_node=node, distance_limit=distance_limit, constructing_phase= constructing_phase)
                # print(crossing)
                if len(crossing) > 1:
                    G_new, G_append = crossing[0], crossing[-1]  #G, p1, Point(X,Y), distance, new_osmid, G_append]
                    if mode == "water":
                        edge_lst.append(node['u'])
                        nodes_added.append(crossing[4])

                        # print(nodes_added)

                

                else:
                    #TODO: somehow save/log why not appended
                    not_connected.append(crossing[0])
            else:
                #save the not connected nodes somewhere?
                not_connected.append(node)

        # might be redundant if the roads are already added through the create_crossing algorithm?        
        # G_airports  = wrg.add_new_roads(inp_pths, G_new)

        #TODO: also save G_append

        ox.io.save_graphml(G_new, filepath=pth_out_small, encoding='utf-8')
        ox.io.save_graphml(G_append, filepath=pth_out_large, encoding='utf-8')
        if mode == "water":
            with open(waterpth+'nodes_added.pk','wb') as f:
                pickle.dump(nodes_added, f)
            with open(waterpth+'oversides.pk','wb') as f:
                pickle.dump(edge_lst, f)
    
    if mode == "water":
        return G_new, G_append, edge_lst, nodes_added

    else:       
        return G_new, G_append




"""

def create_crossing_mp(G_split, X, Y, osmid_in=None, mode=None, G_append = None, edge_attr=None, new_strt_node=None, distance_limit = None):
    G=G_split.copy()
    edge_node_dct = defaultdict(list)
    # print(G.nodes)
    #TODO: project the things first
    edge, distance = ox.distance.nearest_edges(G, X, Y, return_dist = True)
    
    # G[u][v]['color']
    #get nearest edge
    #then get nearest point on edge
    # then use algorithms below to split edge
    print(f'edge: {edge}')
    u,v,i = edge


    try:
        print("trying")

        edge_geom = G[u][v][i]['geometry']
        edge_attrs = G[u][v][i] # Copy edge attributes
        # print(G.nodes[u], G.nodes[v])

        # coords = [Point(x,y)for x,y in edge.coords]

        # The points are returned in the same order as the input geometries:
        # dont know if correct or if needs to be projected first??

        p1, p2 = nearest_points(edge_geom, Point(X,Y))
        # x, y = [x for x in p1.coords]
        osmid =G[u][v][i]['osmid']
        node_type = G[u][v][i]['type']

        
        point_list = [p1]

        


        segments, bool =split_line_with_points(edge_geom, point_list)
        count=1
        print(edge_attrs)
        if (mode == "airstrip") or (mode == "water"):

            #TODO: check if X,Y is correct
            line_geo = gpd.GeoSeries(LineString([p1, Point(X,Y)]), crs='EPSG:4326')
            distance = line_geo.to_crs('EPSG:21897').length[0]

            if distance > distance_limit:
                print(f'too far away {distance}')
                return [osmid_in]



        if( mode == 'new node') or (mode == 'airstrip'):
            print('new start node added')
            if new_strt_node['goal_id'] == new_strt_node['u']:
                osmid_in = osmid_in+new_strt_node['goal_id']
            
            edge_node_dct['nodes'].add({"id": osmid_in, "y": new_strt_node['y_start'], "x": new_strt_node['x_start'],"type": new_strt_node['type']})
            

            # G_append.add_node(osmid_in, y= new_strt_node['y_start'], x= new_strt_node['x_start'], type=new_strt_node['type'])
            # G.add_node(osmid_in, y= new_strt_node['y_start'], x= new_strt_node['x_start'], type=new_strt_node['type'])

        
        #bool that states whether line can be split 
        if bool == True:
            
            #TODO: check if this is correct for all modes
            new_osmid = int(abs(p1.y+p1.x)*1e6)

            edge_node_dct['nodes'].add({"id": new_osmid, "y": p1.y, "x": p1.x, "type": node_type})


            
            # G_append.add_node(new_osmid, y= p1.y, x= p1.x, type=node_type)
            # G.add_node(new_osmid, y= p1.y, x= p1.x, type=node_type)

            for i in segments:
                #TODO: check if with adding edge attributes this still works 
                line_geo = gpd.GeoSeries(i, crs='EPSG:4326')

                length = line_geo.to_crs('EPSG:21897').length[0]
                if count==1:
                    edge_node_dct['edges'].add({"u": u , "v": v, "osmid": new_osmid, **edge_attrs, 'length': length, 'geometry': i} )
                    # G_append.add_edge(u,new_osmid, **{**edge_attrs, 'length': length, 'geometry': i} )
                    # G.add_edge(u,new_osmid, **{**edge_attrs, 'length': length, 'geometry': i} )
                else:
                    edge_node_dct['edges'].add({"u": new_osmid , "v": v, "osmid": new_osmid, **edge_attrs, 'length': length, 'geometry': i} )
                    # G_append.add_edge(new_osmid,v, **{**edge_attrs, 'length': length, 'geometry': i} )
                    # G.add_edge(new_osmid,v, **{**edge_attrs, 'length': length, 'geometry': i} )
                count+=1

            #if water==True than there is an edge added between a road node and a (new) water node
            #TODO: create modes, also one for adding new point and road
            if (mode == "water") or (mode == "airstrip" ):
                print(f'mode is water, edge should be added in theory, edge attr: {edge_attr}')
                edge_node_dct['edges'].add({"u": osmid_in, "v": new_osmid, "osmid": [osmid_in, new_osmid], "length" :  distance, "geometry" = LineString([Point(X,Y),p1]})
                # G_append.add_edge(osmid_in, new_osmid, osmid = [osmid_in, new_osmid], length = distance, geometry = LineString([Point(X,Y),p1]), type = edge_attr)
                # G.add_edge(osmid_in, new_osmid, osmid = [osmid_in, new_osmid], length = distance, geometry = LineString([Point(X,Y),p1]), type = edge_attr)
            
            if mode == "new node":
                edge_node_dct['edges'].add({"u": osmid_in, "v": new_osmid, "osmid": [osmid_in, new_osmid], "length" :  new_strt_node['length'], "geometry" = new_strt_node['geometry'])

                # G_append.add_edge(osmid_in, new_osmid, osmid = [osmid_in, new_osmid], length =new_strt_node['length'], geometry = new_strt_node['geometry'], type = edge_attr)
                # G.add_edge(osmid_in, new_osmid, osmid = [osmid_in, new_osmid], length =new_strt_node['length'], geometry = new_strt_node['geometry'], type = edge_attr)


            
            print(u,v)
            G.remove_edge(*edge)
            print(f'G normal removed edge {u}, {v}')

            G_append.remove_edge(*edge)
            print(f'G_apppend remove edge {u}, {v}')


            return  [G, p1, Point(X,Y), distance, new_osmid, G_append]
       
        #TODO: bool does not mean that there is no connection, but it means that line is not split but connection is to end node
        # or it means that connection is already there ??
        else:
            #TODO: causes key error when combining water and roads

            geometries = nx.get_node_attributes(G_append, "y")
            print(f'geometries: {u}, {v}, {geometries}')

            if geometries[u] ==  p1.y:
                p_goal = edge_geom.coords[0]
                new_osmid = u

            elif geometries[v] ==  p1.y:
                p_goal = edge_geom.coords[-1]
                new_osmid = v

            else:
                print('no match')
                return [osmid_in]




            #if water==True than there is an edge added between a road node and a (new) water node
            #TODO: create modes, also one for adding new point and road


            if (mode == "water") or (mode == "airstrip" ):
                G_append.add_edge(osmid_in, new_osmid, osmid = [osmid_in, new_osmid], length = distance, geometry = LineString([Point(X,Y),p_goal]), type = edge_attr)
                G_append.add_edge(osmid_in, new_osmid, osmid = [osmid_in, new_osmid], length = distance, geometry = LineString([Point(X,Y),p_goal]), type = edge_attr)
            elif (mode == "new_node"):
                G_append.add_edge(osmid_in, new_osmid, osmid = [osmid_in, new_osmid], length =new_strt_node['length'], geometry = new_strt_node['geometry'], type = edge_attr)
                G_append.add_edge(osmid_in, new_osmid, osmid = [osmid_in, new_osmid], length =new_strt_node['length'], geometry = new_strt_node['geometry'], type = edge_attr)
            else:
                print('still no match')


            return  [G, p1, Point(X,Y), distance, new_osmid, G_append]
    except Exception:
        traceback.print_exc()             
        return [osmid_in]

"""