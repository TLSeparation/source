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
__copyright__ = "Copyright 2017, TLSeparation Project"
__credits__ = ["Matheus Boni Vicari"]
__license__ = "GPL3"
__version__ = "1.2.1.1"
__maintainer__ = "Matheus Boni Vicari"
__email__ = "matheus.boni.vicari@gmail.com"
__status__ = "Development"

import numpy as np
from sklearn.neighbors import NearestNeighbors


def continuity_filter_new(arr, base_id, kpairs, knn, nbrs_threshold):

    """
    Converts a numpy.array of points coordinates into a Weighted BiDirectional
    NetworkX Graph.
    This funcions uses a NearestNeighbor search to determine points adajency.
    The NNsearch results are used to select pairs of points (or nodes) that
    have a common edge.


    Args:
        arr (array): n-dimensional array of points.
        base_id (int): index of base id (root) in the graph.
        kpairs (int): number of points around each point in arr to select in
            order to build edges.
        knn (int): Number of neighbors to search around each point in the
            neighborhood phase. The higher the better (careful, it's  memory
            intensive).
        nbrs_threshold (float): Maximum valid distance between neighbors
            points.
        nbrs_threshold_step (float): Distance increment used in the final
            phase of edges generation. It's used to make sure that in the
            end, every point in arr will be translated to nodes in the graph.
        graph_threshold (float): Maximum distance between pairs of nodes
            (edge distance) accepted in the graph generation.

    Returns:
        G (networkx graph): Graph containing all points in 'arr' as nodes.

    """

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
                for ni, di in zip(nn_idx, dd_idx):
                    if di <= nbrs_threshold:
                        nntemp.append(ni)

            # Obtaining an unique array of points currently being processed.
            current_idx = np.unique(nntemp)

        # If current_idx is an empty list.
        elif len(current_idx) == 0:
            break

        # Appending current_idx to processed_idx.
        processed_idx = np.append(processed_idx, current_idx)
        processed_idx = np.unique(processed_idx).astype(int)

        # Generating list of remaining proints to process.
        idx = idx_base[np.in1d(idx_base, processed_idx, invert=True)]

    return processed_idx
