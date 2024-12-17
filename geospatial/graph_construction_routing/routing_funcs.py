import pickle
import pandas as pd
import glob
import networkx as nx

def _build_paths_from_predecessors(sources, target, pred):
    """Compute all simple paths to target, given the predecessors found in
    pred, terminating when any source in sources is found.

    Parameters
    ----------
    sources : set
       Starting nodes for path.

    target : node
       Ending node for path.

    pred : dict
       A dictionary of predecessor lists, keyed by node

    Returns
    -------
    paths : generator of lists
        A generator of all paths between source and target.

    Raises
    ------
    NetworkXNoPath
        If `target` cannot be reached from `source`.

    Notes
    -----
    There may be many paths between the sources and target.  If there are
    cycles among the predecessors, this function will not produce all
    possible paths because doing so would produce infinitely many paths
    of unbounded length -- instead, we only produce simple paths.

    See Also
    --------
    shortest_path
    single_source_shortest_path
    all_pairs_shortest_path
    all_shortest_paths
    bellman_ford_path
    """
    if target not in pred:
        raise nx.NetworkXNoPath(f"Target {target} cannot be reached from given sources")

    seen = {target}
    stack = [[target, 0]]
    top = 0
    while top >= 0:
        node, i = stack[top]
        if node in sources:
            yield [p for p, n in reversed(stack[: top + 1])]
        if len(pred[node]) > i:
            stack[top][1] = i + 1
            next = pred[node][i]
            if next in seen:
                continue
            else:
                seen.add(next)
            top += 1
            if top == len(stack):
                stack.append([next, 0])
            else:
                stack[top][:] = [next, 0]
        else:
            seen.discard(node)
            top -= 1
    

def select_pths(data_pth):
    fields_labs_shortest = []
    df_pths =[]
    df_pths.extend(glob.glob(data_pth))
    fields_paths = []

    pths_df = pd.DataFrame(columns=['source','target', 'path', 'distance'])

    for pth in df_pths:
        with open(pth,'rb') as f:
            df = pickle.load(f)
            pths_df = pd.concat([df, pths_df], ignore_index=True)

    for source in pths_df.source.unique():
        pths_df_sorted = pths_df.loc[pths_df['source']== source].sort_values(by='distance')
        fields_labs_shortest.append([source, pths_df_sorted.iloc[0]['target']])
        fields_paths.append(pths_df_sorted.iloc[0]['path'])
    return fields_labs_shortest, fields_paths
        
    


def dijkstra_mp(S0_un, weight, startstop, name):
 
    count =0 
    save_counter=0

    pths_df = pd.DataFrame(columns=['source','target', 'path', 'distance'])
    for i in range(len(startstop)):
        pred, dist = nx.dijkstra_predecessor_and_distance(S0_un, startstop[i][0], weight=weight)

        pth = _build_paths_from_predecessors([startstop[i][0]], startstop[i][1], pred)
        
        for p in pth:
            path = p
        
        # pths_df.add( {'source': startstop[i][0] ,'target': startstop[i][1], 'path': path, 'distance': dist[startstop[i][1]]})
        pths_df = pd.concat([pd.DataFrame([[startstop[i][0] , startstop[i][1], path, dist[startstop[i][1]]]], columns=pths_df.columns), pths_df], ignore_index=True)
        count+=1
        if count == 2500:
            with open(f'./{name}_{weight}{save_counter}.pk','wb') as f:
                pickle.dump(pths_df, f)
                pths_df = pd.DataFrame(columns=['source','target', 'path', 'distance'])

            count=0
            save_counter+=1
    with open(f'./{name}_{weight}{save_counter}.pk','wb') as f:
        pickle.dump(pths_df, f)
        pths_df = pd.DataFrame(columns=['source','target', 'path', 'distance'])

def add_paths_to_graph(G, pths, clip):
    combined_single_paths = []
    count=0
    G= G.copy()
    for path in pths:
        pairs = [path[i: i + 2] for i in range(len(path)-1)]
        combined_single_paths.extend(pairs)

    for pth in combined_single_paths:
        count+=1
        try:
            # print(G_tt[pth[0]][pth[1]])
            G[pth[0]][pth[1]][0]['count'] +=1
        except:
            try:   
                G[pth[1]][pth[0]][0]['count'] +=1
            except:
                try:
                    G[pth[0]][pth[1]][1]['count'] +=1
                except:
                    
                    G[pth[1]][pth[0]][1]['count'] +=1
    print(count)
    if clip == True:
        attr = nx.get_edge_attributes(G, 'count')
        node_lst=[k for k,v in attr.items() if v > 0]

                        
        G = G.edge_subgraph(node_lst).copy()


    return G