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
from shortpath import (array_to_graph, extract_path_info)
from sklearn.neighbors import NearestNeighbors
from data_utils import get_diff
import hdbscan


def path_clustering(arr, knn, slice_length, cluster_threshold,
                    growth_radius, freq_threshold=0.6, max_diam=1):

    """
    Generates the path clustering of a point cloud with a defined
    root point.

    Args:
        arr (array): N-dimensional array (m x n) containing a set of
            parameters (n) over a set of observations (m). In this case, the
            set of parameters are the point cloud coordinates, where each row
            represents a point.
        knn (int): Number of nearest neighbors to use in the reconstruction of
            point cloud around the generated path nodes. A high value may lead
            to unnecessary duplication of steps. A low value may lead to gaps
            in the reconstructed point cloud.
        slice_length (float): Length of the slices of the data in 'arr'.
        cluster_threshold (float): Distance threshold to be used as constraint
            in the slice clustering step.

    Returns:
        wood (array): N-dimensional array (w x n) containing the (w) points
            classified as wood from the path reconstruction. The columns (n)
            represents the   3D coordinates of each point.
        leaf (array): N-dimensional array (l x n) containing the (l) points
            not classified as wood and, therefore, classified as leaf. The
            columns (n) represents the 3D coordinates of each point.

    """

    # Slicing and clustering the data to generate the center points of
    # every cluster. Return also the cluster data and the diameter of
    # each cluster.
    nodes_data, nodes_diameter, _, _ = slice_nodes(arr, slice_length,
                                                   cluster_threshold)

    # Obtaining the central nodes coordinates (tree skeleton points) and
    # their respective diameters.
    central_nodes = np.asarray(nodes_data.keys())
    diameter = np.asarray(nodes_diameter.values())

    # Filtering central_nodes by a maximum diameter to prevent clusters too
    # large affecting the segmentation.
    mask = diameter <= max_diam
    central_nodes = central_nodes[mask]

    # Detecting base point of central_nodes.
    base_point = np.argmin(central_nodes[:, 2])

    # Calculating the shortest path over the central nodes.
    G = array_to_graph(central_nodes, base_point, 3, 100, 0.05, 0.02, 0.5)
    nodes_ids, dist, path = extract_path_info(G, base_point,
                                              return_path=True)
    gnodes = central_nodes[nodes_ids]
    gdist = np.array(dist)

    # Extracting all the nodes in the shortest path.
    gpath = path.values()
    gpath_nodes = [i for j in gpath for i in j]

    # Obtaining all unique values in the central nodes path and their
    # respective frequency.
    gpath_nodes, freq = np.unique(gpath_nodes, return_counts=True)

    # Log transforming the frequency values.
    freq_log = np.log(freq)

    # Filtering central nodes based on the frequency of paths passing by
    # each node.
    gp = gnodes[freq_log >= (np.max(freq_log) * freq_threshold)]

    # Obtaining list of close nodes that are not yet in 'gp' and stacking them
    # to 'gp'. This step aims to fill the gaps between nodes from 'gp'.
    nbrs = NearestNeighbors(leaf_size=15, n_jobs=-1).fit(gnodes)

    # Getting list of neares neighbors to each point in gp.
    idx = nbrs.kneighbors(gp, n_neighbors=knn, return_distance=False)
    idx = np.unique(idx)

    # Stacking gp with new neighboring indices.
    gp = np.vstack((gp, gnodes[idx]))
    # Initializing array of selected points 'pts'.
    pts = gp

    # Calculating number of points in gp.
    npw = gp.shape[0]

    # Setting initial diffrence value (e) and difference threhsold
    # (e_threshold). These values will be used to detect when to stop the
    # iterative process of filling gaps.
    e = 9999999
    e_threshold = 10

    # While the new number of points are smaller than threshold.
    while e > e_threshold:
        # Get new neighbors current set of selected points (pts).
        idx = nbrs.radius_neighbors(pts, radius=growth_radius,
                                    return_distance=False)

        # Initializing list of possible new points and filtering them by
        # their shortest path distance. This criteria aims to select only
        # points that are closer to base than current points (pts).
        id1 = []
        for i in idx:
            id1.append(i[1:][gdist[i[1:]] <= gdist[i[0]]])
        id1 = np.unique([j for i in id1 for j in i])

        # Stacking new points to 'gp' and getting list of points that remain
        # to be processed (pts).
        pts = get_diff(gp, gnodes[id1])
        gp = np.vstack((gp, pts))

        # Calculating step difference.
        e = gp.shape[0] - npw
        npw = gp.shape[0]

    # Obtaining the data from each respective node in 'gp'. In this case the
    # data from the skeleton nodes are considered as wood and the remaining
    # data is set as leaf.
    try:
        keys = tuple(map(tuple, gp))
        vals = map(nodes_data.get, keys)
        vals = filter(lambda v: v is not None, vals)
        wood = np.concatenate(vals, axis=0)

        leaf = get_diff(arr, wood)

        return wood, leaf
    except:
        return [], []


def slice_nodes(arr, slice_length, cluster_threshold):

    """
    Generates the skeleton points of a 3D point cloud by slicing it
    based on the shortest path distance of every point from the base.


    Args:
        arr (array): N-dimensional array (m x n) containing a set of
            parameters (n) over a set of observations (m). In this case, the
            set of parameters are the point cloud coordinates, where each row
            represents a point.
        slice_length (float): Length for the slices of data in 'arr'.
        cluster_threshold (float): Distance threshold to be used as
            constraint in the slice clustering step.

    Returns:
        cluster_data (dict): Dictionary containing the skeleton nodes
            coordinates (keys) and the substet of points from 'arr' that
            generated the respective skeleton nodes.
        cluster_diameter (dict): Dictionary containing the skeleton nodes
            coordinates (keys) and the mean diameter of the cluster that
            generated the respective skeleton nodes.

    """

    # Calculating the shortest path distance for the input array (arr).
    # Here, the calculate_path module is called twice in order to reach
    # as many points in 'arr' as possible.
    G = array_to_graph(arr, 0, 3, 100, 0.05, 0.02, 0.5)
    nodes_ids, dist = extract_path_info(G, 0, return_path=False)
    nodes = arr[nodes_ids]

    # Initializing the dictionary variables for output.
    cluster_data = dict()
    cluster_diameter = dict()

    # Generating the indices of each slice.
    slice_id = np.round((dist - np.min(dist, axis=0)) /
                        slice_length).astype(int)

    # Looping over each slice of the data.
    for i in np.unique(slice_id):
        # Selecting the data from the current slice.
        data_slice = nodes[i == slice_id]

        try:
            # Clustering the data.
            clusters = data_clustering(data_slice, cluster_threshold)
            # Looping over every cluster in the slice.
            for j in np.unique(clusters):
                # Selecting data from the current cluster.
                d = data_slice[j == clusters]

                # Calculating the central coord and diameter of the current
                # cluster.
                center, diameter = central_coord(d)
                xm, ym, zm = center
                cluster_data[xm, ym, zm] = d
                cluster_diameter[xm, ym, zm] = diameter

        except:
            pass

    return cluster_data, cluster_diameter, nodes, dist


def central_coord(arr):

    """
    Calculates the central coordinates and mean diameter of an array of
    points in 3D space.

    Args:
        arr (array): N-dimensional array (m x n) containing a set of
            parameters (n) over a set of observations (m). In this case, the
            set of parameters are the point cloud coordinates, where each
            row represents a point.

    Returns:
        coord (array): Central coordinates of the points in the input array.
        diameter (float): Mean diameter of the points in the input array.

    """

    # Calculating extents of 'arr' in every dimension.
    min_ = np.min(arr, axis=0)
    max_ = np.max(arr, axis=0)

    # Calculates central coordinate and mean range (diameter).
    return min_ + ((max_ - min_) / 2), np.mean(max_[:2] - min_[:2])


def data_clustering(point_arr, threshold):

    """
    Clusters point_arr using hierarchical clustering.

    Args:
        point_arr (array): N-dimensional array (m x n) containing a set of
            parameters (n) over a set of observations (m). In this case, the
            set of parameters are the point cloud coordinates, where each row
            represents a point.
        threshold (float): Distance threshold to be used as constraint in the
            slice clustering step.

    Returns:
        clusters.labels_ (array): Set of cluster labels for the classified
            array.

    """

    clusterer = hdbscan.HDBSCAN()
    clusterer.fit(point_arr)

    return clusterer.labels_
