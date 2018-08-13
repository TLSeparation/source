# Copyright (c) 2017, Matheus Boni Vicari, TLSeparation Project
# All rights reserved.
#
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.


__author__ = "Matheus Boni Vicari"
__copyright__ = "Copyright 2017-2018, TLSeparation Project"
__credits__ = ["Matheus Boni Vicari"]
__license__ = "GPL3"
__version__ = "1.3"
__maintainer__ = "Matheus Boni Vicari"
__email__ = "matheus.boni.vicari@gmail.com"
__status__ = "Development"


import networkx as nx
import numpy as np
from sklearn.neighbors import NearestNeighbors


def array_to_graph(arr, base_id, kpairs, knn, nbrs_threshold,
                   nbrs_threshold_step, graph_threshold=np.inf):

    """
    Converts a numpy.array of points coordinates into a Weighted BiDirectional
    NetworkX Graph.
    This funcions uses a NearestNeighbor search to determine points adajency.
    The NNsearch results are used to select pairs of points (or nodes) that
    have a common edge.


    Parameters
    ----------
    arr : array
        n-dimensional array of points.
    base_id : int
        Index of base id (root) in the graph.
    kpairs : int
        Number of points around each point in arr to select in order to
        build edges.
    knn : int
        Number of neighbors to search around each point in the neighborhood
        phase. The higher the better (careful, it's  memory intensive).
    nbrs_threshold : float
        Maximum valid distance between neighbors points.
    nbrs_threshold_step : float
        Distance increment used in the final phase of edges generation. It's
        used to make sure that in the end, every point in arr will be
        translated to nodes in the graph.
    graph_threshold : float
        Maximum distance between pairs of nodes (edge distance) accepted in
        the graph generation.

    Returns
    -------
    G : networkx graph
        Graph containing all points in 'arr' as nodes.

    """

    # Initializing graph.
    G = nx.Graph()

    # Generating array of all indices from 'arr' and all indices to process
    # 'idx'.
    idx_base = np.arange(arr.shape[0], dtype=int)
    idx = np.arange(arr.shape[0], dtype=int)

    # Initializing NearestNeighbors search and searching for all 'knn'
    # neighboring points arround each point in 'arr'.
    nbrs = NearestNeighbors(n_neighbors=knn, metric='euclidean',
                            leaf_size=15, n_jobs=-1).fit(arr)
    distances, indices = nbrs.kneighbors(arr)
    indices = indices.astype(int)

    # Initializing variables for current ids being processed (current_idx)
    # and all ids already processed (processed_idx).
    current_idx = [base_id]
    processed_idx = [base_id]

    # Looping while there are still indices (idx) left to process.
    while idx.shape[0] > 0:

        # If current_idx is a list containing several indices.
        if len(current_idx) > 0:

            # Selecting NearestNeighbors indices and distances for current
            # indices being processed.
            nn = indices[current_idx]
            dd = distances[current_idx]

            # Masking out indices already contained in processed_idx.
            mask1 = np.in1d(nn, processed_idx, invert=True).reshape(nn.shape)

            # Initializing temporary list of nearest neighbors. This list
            # is latter used to accumulate points that will be added to
            # processed points list.
            nntemp = []

            # Looping over current indices's set of nn points and selecting
            # knn points that hasn't been added/processed yet (mask1).
            for i, (n, d, g) in enumerate(zip(nn, dd, current_idx)):
                nn_idx = n[mask1[i]][0:kpairs+1]
                dd_idx = d[mask1[i]][0:kpairs+1]
                nntemp.append(nn_idx)

                # Adding current knn selected points as nodes to graph G.
                add_nodes(G, g, nn_idx, dd_idx, graph_threshold)

            # Obtaining an unique array of points currently being processed.
            current_idx = np.unique([t2 for t1 in nntemp for t2 in t1])

        # If current_idx is an empty list.
        elif len(current_idx) == 0:

            # Getting NearestNeighbors indices and distance for all indices
            # that remain to be processed.
            idx2 = indices[idx]
            dist2 = distances[idx]

            # Masking indices in idx2 that have already been processed. The
            # idea is to connect remaining points to existing graph nodes.
            mask1 = np.in1d(idx2, processed_idx).reshape(idx2.shape)
            # Masking neighboring points that are withing threshold distance.
            mask2 = dist2 < nbrs_threshold
            # mask1 AND mask2. This will mask only indices that are part of
            # the graph and within threshold distance.
            mask = np.logical_and(mask1, mask2)

            # Getting unique array of indices that match the criteria from
            # mask1 and mask2.
            temp_idx = np.unique(np.where(mask)[0])
            # Assigns remaining indices (idx) matched in temp_idx to
            # current_idx.
            current_idx = idx[temp_idx]

            # Selecting NearestNeighbors indices and distances for current
            # indices being processed.
            nn = indices[current_idx]
            dd = distances[current_idx]

            # Masking points in nn that have already been processed.
            # This is the oposite approach as above, where points that are
            # still not in the graph are desired. Now, to make sure the
            # continuity of the graph is kept, join current remaining indices
            # to indices already in G.
            mask = np.in1d(nn, processed_idx, invert=True).reshape(nn.shape)

            # Initializing temporary list of nearest neighbors. This list
            # is latter used to accumulate points that will be added to
            # processed points list.
            nntemp = []

            # Looping over current indices's set of nn points and selecting
            # knn points that have alreay been added/processed (mask).
            # Also, to ensure continuity over next iteration, select another
            # kpairs points from indices that haven't been processed (~mask).
            for i, (n, d, g) in enumerate(zip(nn, dd, current_idx)):
                nn_idx = n[mask[i]][0:kpairs+1]
                dd_idx = d[mask[i]][0:kpairs+1]

                # Adding current knn selected points as nodes to graph G.
                add_nodes(G, g, nn_idx, dd_idx, graph_threshold)

                nn_idx = n[~mask[i]][0:kpairs+1]
                dd_idx = d[~mask[i]][0:kpairs+1]

                # Adding current knn selected points as nodes to graph G.
                add_nodes(G, g, nn_idx, dd_idx, graph_threshold)

            # Check if current_idx is still empty. If so, increase the
            # nbrs_threshold to try to include more points in the next
            # iteration.
            if len(current_idx) == 0:
                nbrs_threshold += nbrs_threshold_step

        # Appending current_idx to processed_idx.
        processed_idx = np.append(processed_idx, current_idx)
        processed_idx = np.unique(processed_idx).astype(int)

        # Generating list of remaining proints to process.
        idx = idx_base[np.in1d(idx_base, processed_idx, invert=True)]

    return G


def extract_path_info(G, base_id, return_path=True):

    """
    Extracts shortest path information from a NetworkX graph.

    Parameters
    ----------
    G : networkx graph
        NetworkX graph object from which to extract the information.
    base_id : int
        Base (root) node id to calculate the shortest path for all other
        nodes.
    return_path : boolean
        Option to select if function should output path list for every node
        in G to base_id.

    Returns
    -------
    nodes_ids : list
        Indices of all nodes in graph G.
    distance : list
        Shortest path distance (accumulated) from all nodes in G to base_id
        node.
    path_list : dict
        Dictionary of nodes that comprises the path of every node in G to
        base_id node.

    """

    # Calculating the shortest path
    shortpath = nx.single_source_dijkstra_path_length(G, base_id)

    # Obtaining the node coordinates and their respective distance from
    # the base point.
    nodes_ids = shortpath.keys()
    distance = shortpath.values()

    # Checking if the function should also return the paths of each node and
    # if so, generating the path list and returning it.
    if return_path is True:
        path_list = nx.single_source_dijkstra_path(G, base_id)
        return nodes_ids, distance, path_list

    elif return_path is False:
        return nodes_ids, distance


def add_nodes(G, base_node, indices, distance, threshold):

    """
    Adds a set of nodes and weighted edges based on pairs of indices
    between base_node and all entries in indices. Each node pair shares an
    edge with weight equal to the distance between both nodes.

    Parameters
    ----------
    G : networkx graph
        NetworkX graph object to which all nodes/edges will be added.
    base_node : int
        Base node's id to be added. All other nodes will be paired with
        base_node to form different edges.
    indices : list or array
        Set of nodes indices to be paired with base_node.
    distance : list or array
        Set of distances between all nodes in 'indices' and base_node.
    threshold : float
        Edge distance threshold. All edges with distance larger than
        'threshold' will not be added to G.

    """

    for c in np.arange(len(indices)):
        if distance[c] <= threshold:
            # If the distance between vertices is less than a given
            # threshold, add edge (i[0], i[c]) to Graph.
            G.add_weighted_edges_from([(base_node, indices[c],
                                        distance[c])])
