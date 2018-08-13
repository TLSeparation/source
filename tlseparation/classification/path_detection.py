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
__credits__ = ["Matheus Boni Vicari", "Phil Wilkes"]
__license__ = "GPL3"
__version__ = "1.3"
__maintainer__ = "Matheus Boni Vicari"
__email__ = "matheus.boni.vicari@gmail.com"
__status__ = "Development"


import datetime
import numpy as np
from sklearn.neighbors import NearestNeighbors
from ..utility.shortpath import (array_to_graph, extract_path_info)
from ..utility.voxels import voxelize_cloud
from ..utility.downsampling import (downsample_cloud, upsample_cloud)
from ..utility.filtering import radius_filter
from ..utility.knnsearch import set_nbrs_rad


def path_detect_frequency(point_cloud, downsample_size,
                          frequency_threshold):

    """
    Detects points from major paths in a graph generated from a point cloud.
    The detection is performed by comparing the frequency of all paths that
    each node is present. Nodes with frequency larger than threshold are
    selected as detected. In order to fill pathways regions with low nodes
    density, neighboring points within downsampling_size * 1.5 distance are
    also set as detected.

    Parameters
    ----------
    point_cloud : numpy.ndarray
        2D (n x 3) array containing n points in 3D space (x, y, z).
    downsample_size : float
        Distance threshold used to group (downsample) the input point cloud.
        Simplificaton of the cloud by downsampling, improves the results and
        processing times.
    frequency_threshold : float
        Minimum path frequency for a node to be selected as part of major
        pathways.

    Returns
    -------
    path_points: numpy.ndarray
        2D (np x 3) array containing n points in 3D space (x, y, z) that
        belongs to major pathways in the point cloud.

    """

    # Downsampling point cloud. The function returns downsampled indices
    # (down_ids) and a set of original neighboring indices around each
    # downsampled points (up_ids) that can be later used to revert the
    # downsampling.
    down_ids, up_ids = downsample_cloud(point_cloud, downsample_size,
                                        return_indices=True,
                                        return_neighbors=True)
    # Obtaining downsampled cloud base index (lowest point in the cloud).
    base_id = np.argmin(point_cloud[down_ids, 2])
    # Generating networkx graph from point cloud.
    G = array_to_graph(point_cloud[down_ids], base_id, 3, 100,
                       downsample_size * 1.77, 0.02)
    # Extracting shortest path information from graph: nodes indices
    # (nodes_ids), shortest path distance (D) and list of nodes in each
    # nodes' paths (path_dict).
    nodes_ids, D, path_dict = extract_path_info(G, base_id,
                                                return_path=True)
    # Selecting nodes coordinates
    nodes = point_cloud[down_ids][nodes_ids]
    # Unpacking all indices in path_dict and appending them to path_ids_list.
    path_ids_list = []
    for k, v in path_dict.iteritems():
        path_ids_list.append(v)
    # Flattening path_ids_list.
    path_ids_list = [j for i in path_ids_list for j in i]

    # Calculating counts of paths (c) that go through each unique node (u).
    u, c = np.unique(path_ids_list, return_counts=True)
    # Masking nodes with log(c) larger than treshold.
    mask = np.log(c) >= frequency_threshold
    # Filtering isolated nodes.
    mask_radius = radius_filter(nodes[mask], 0.2, 3)
    # Selecting neighboring nodes.
    nbrs_idx = set_nbrs_rad(point_cloud, nodes[mask][mask_radius],
                            downsample_size * 1.5, False)

    # Flattening list of neighboring nodes and upscaling results to
    # original point cloud.
    ids = [j for i in nbrs_idx for j in i]
    ids = np.unique(ids)
    ups_ids = upsample_cloud(ids, up_ids)

    return point_cloud[ups_ids]


def voxel_path_detection(point_cloud, voxel_size, k_retrace, knn,
                         nbrs_threshold, verbose=False):

    """
    Applies detect_main_pathways but with a voxelization option to speed up
    processing.

    Parameters
    ----------
    point_cloud : array
        Three-dimensional point cloud of a single tree to perform the
        wood-leaf separation. This should be a n-dimensional array (m x n)
        containing a set of coordinates (n) over a set of points (m).
    voxel_size: float
        Voxel dimensions' size.
    k_retrace : int
        Number of steps in the graph to retrace back to graph's base. Every
        node in graph will be moved  k_retrace steps from the extremities
        towards to base.
    knn : int
        Number of neighbors to fill gaps in detected paths. The larger the
        better. A large knn will increase memory usage. Recommended value
        between 50 and 150.
    nbrs_threshold : float
        Maximum distance to valid neighboring points used to fill gaps in
        detected paths.
    verbose: bool
        Option to set verbose on/off.

    Returns
    -------
    path_mask : array
        Boolean mask where 'True' represents points detected as part of the
        main pathways and 'False' represents points not part of the pathways.

    Raises
    ------
    AssertionError:
        point_cloud has the wrong shape or number of dimensions.
    """

    # Making sure input point cloud has the right shape and number of
    # dimensions.
    assert point_cloud.ndim == 2, "point_cloud must be an array with 2\
 dimensions, n_points x 3 (x, y, z)."
    assert point_cloud.shape[1] == 3, "point_cloud must be a 3D point cloud.\
 Make sure it has the shape n_points x 3 (x, y, z)."

    # Voxelizing point cloud.
    if verbose:
        print(str(datetime.datetime.now()) + ' | >>> voxelizing point cloud, \
with a voxel size of %s' % voxel_size)
    vox = voxelize_cloud(point_cloud, voxel_size=voxel_size)
    vox_coords = np.asarray(vox.keys())

    # Running detect_main_pathways over voxels' coordinates.
    if verbose:
        print(str(datetime.datetime.now()) + ' | >>> running \
detect_main_pathways with %s number of steps retraced' % k_retrace)
    path_mask_voxel = detect_main_pathways(vox_coords, k_retrace, knn,
                                           nbrs_threshold, verbose=verbose)
    # Re-indexing point_cloud indices from voxels coordinates detected as
    # part of the path.
    path_ids = np.unique([j for i in vox_coords[path_mask_voxel] for
                          j in vox[tuple(i)]])
    path_mask = np.zeros(point_cloud.shape[0], dtype=bool)
    path_mask[path_ids] = True

    return path_mask


def detect_main_pathways(point_cloud, k_retrace, knn, nbrs_threshold,
                         verbose=False, max_iter=100):

    """
    Detects the main pathways of an unordered 3D point cloud. Set as true
    all points detected as part of all detected pathways that down to the
    base of the graph.

    Parameters
    ----------
    point_cloud : array
        Three-dimensional point cloud of a single tree to perform the
        wood-leaf separation. This should be a n-dimensional array (m x n)
        containing a set of coordinates (n) over a set of points (m).
    k_retrace : int
        Number of steps in the graph to retrace back to graph's base. Every
        node in graph will be moved  k_retrace steps from the extremities
        towards to base.
    knn : int
        Number of neighbors to fill gaps in detected paths. The larger the
        better. A large knn will increase memory usage. Recommended value
        between 50 and 150.
    nbrs_threshold : float
        Maximum distance to valid neighboring points used to fill gaps in
        detected paths.
    verbose: bool
        Option to set verbose on/off.

    Returns
    -------
    path_mask : array
        Boolean mask where 'True' represents points detected as part of the
        main pathways and 'False' represents points not part of the pathways.

    Raises
    ------
    AssertionError:
        point_cloud has the wrong shape or number of dimensions.

    """

    # Making sure input point cloud has the right shape and number of
    # dimensions.
    assert point_cloud.ndim == 2, "point_cloud must be an array with 2\
 dimensions, n_points x 3 (x, y, z)."
    assert point_cloud.shape[1] == 3, "point_cloud must be a 3D point cloud.\
 Make sure it has the shape n_points x 3 (x, y, z)."

    # Getting root index (base_id) from point cloud.
    base_id = np.argmin(point_cloud[:, 2])

    # Generating graph from point cloud and extracting shortest path
    # information.
    if verbose:
        print(str(datetime.datetime.now()) + ' | >>> generating graph from \
point cloud and extracting shortest path information')
    G = array_to_graph(point_cloud, base_id, 3, knn, nbrs_threshold, 0.02)
    nodes_ids, D, path_list = extract_path_info(G, base_id,
                                                return_path=True)
    # Obtaining nodes coordinates from shortest path information.
    nodes = point_cloud[nodes_ids]
    # Converting list of shortest path distances to array.
    D = np.asarray(D)

    # Retracing path for nodes in G. This step aims to detect only major
    # pathways in G. For a tree, these paths are expected to represent
    # branches and trunk.
    new_id = np.zeros(nodes.shape[0], dtype='int')
    for key, values in path_list.iteritems():
        if len(values) >= k_retrace:
            new_id[key] = values[len(values) - k_retrace]
        else:
            new_id[key] = values[0]

    # Getting unique indices after retracing path_list.
    ids = np.unique(new_id)

    # Generating array of all indices from 'arr' and all indices to process
    # 'idx'.
    idx_base = np.arange(point_cloud.shape[0], dtype=int)
    idx = np.arange(point_cloud.shape[0], dtype=int)

    # Initializing NearestNeighbors search and searching for all 'knn'
    # neighboring points arround each point in 'arr'.
    if verbose:
        print(str(datetime.datetime.now()) + ' | >>> initializing \
NearestNeighbors search and searching for all knn neighboring points \
arround each point in arr')
    nbrs = NearestNeighbors(n_neighbors=knn, metric='euclidean',
                            leaf_size=15, n_jobs=-1).fit(point_cloud)
    distances, indices = nbrs.kneighbors(point_cloud)
    indices = indices.astype(int)

    # Initializing variables for current ids being processed (current_idx)
    # and all ids already processed (processed_idx).
    current_idx = ids
    processed_idx = ids

    # Looping while there are still indices in current_idx to process.
    if verbose:
        print(str(datetime.datetime.now()) + ' | >>> looping while there \
are still indices in current_idx to process')
    iteration = 0
    while (len(current_idx) > 0) & (iteration <= max_iter):

        # Selecting NearestNeighbors indices and distances for current
        # indices being processed.
        nn = indices[current_idx]
        dd = distances[current_idx]

        # Masking out indices already contained in processed_idx.
        mask1 = np.in1d(nn, processed_idx, invert=True).reshape(nn.shape)
        # Masking neighboring points that are withing threshold distance.
        mask2 = dd < nbrs_threshold
        # mask1 AND mask2. This will mask only indices that are part of
        # the graph and within threshold distance.
        mask = np.logical_and(mask1, mask2)

        # Initializing temporary list of nearest neighbors. This list
        # is latter used to accumulate points that will be added to
        # processed points list.
        nntemp = []

        # Looping over current indices's set of nn points and selecting
        # knn points that hasn't been added/processed yet (mask1).
        for i, (n, d) in enumerate(zip(nn, dd)):
            nn_idx = n[mask[i]][1:]

            # Checking if current neighbor has an accumulated distance
            # shorter than central node (n[0]) minus some distance based
            # on nbrs_threshold. This penalisation aims to restrict potential
            # neighbors to those more likely to be along an actual path. This
            # would remove points placed along the sides of a path.
            for ni in nn_idx:
                if D[ni] <= D[n[0]] - (nbrs_threshold / 3):
                    nntemp.append(ni)

        # Obtaining an unique array of points currently being processed.
        current_idx = np.unique(nntemp)
        # Updating array of processed indices with indices processed within
        # current iteration (current_idx).
        processed_idx = np.append(processed_idx, current_idx)
        processed_idx = np.unique(processed_idx).astype(int)

        # Generating list of remaining proints to process.
        idx = idx_base[np.in1d(idx_base, processed_idx, invert=True)]

        # Increasing one iteration step.
        iteration += 1

    # Just in case of not having detected all points in the desired paths, run
    # another last iteration.

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
    n_idx = idx[temp_idx]

    # Selecting NearestNeighbors indices and distances for current
    # indices being processed.
    nn = indices[n_idx]
    dd = distances[n_idx]

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
    if verbose:
        print(str(datetime.datetime.now()) + ' | >>> looping over current \
indicess set of nn points and selecting knn points that have alreay been \
added/processed (mask)')
    for i, n in enumerate(nn):
        nn_idx = n[mask[i]][1:]

        # Checking if current neighbor has an accumulated distance
        # shorter than central node (n[0]).
        for ni in nn_idx:
            if D[ni] <= D[n[0]] - (nbrs_threshold / 3):
                nntemp.append(ni)

        nn_idx = n[~mask[i]][1:]

        # Checking if current neighbor has an accumulated distance
        # shorter than central node (n[0]).
        for ni in nn_idx:
            if D[ni] <= D[n[0]] - (nbrs_threshold / 3):
                nntemp.append(ni)

    current_idx = np.unique(nntemp)

    # Appending current_idx to processed_idx.
    processed_idx = np.append(processed_idx, current_idx)
    processed_idx = np.unique(processed_idx).astype(int)

    # Generating list of remaining proints to process.
    idx = idx_base[np.in1d(idx_base, processed_idx, invert=True)]

    # Generating final path mask and setting processed indices as True.
    path_mask = np.zeros(point_cloud.shape[0], dtype=bool)
    path_mask[processed_idx] = True

    return path_mask


def get_base(point_cloud, base_height):

    """
    Get the base of a point cloud based on a certain height from the bottom.

    Parameters
    ----------
    point_cloud : array
        Three-dimensional point cloud of a single tree to perform the
        wood-leaf separation. This should be a n-dimensional array (m x n)
        containing a set of coordinates (n) over a set of points (m).
    base_height : float
        Height of the base slice to mask.

    Returns
    -------
    mask : array
        Base slice masked as True.

    """

    return point_cloud[:, 2] <= base_height
