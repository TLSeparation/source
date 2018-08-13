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

import numpy as np
from knnsearch import (set_nbrs_knn, set_nbrs_rad)
from data_utils import (get_diff, remove_duplicates)
from shortpath import (array_to_graph, extract_path_info)
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from ..classification.point_features import (svd_evals, knn_features,
                                             curvature)


def feature_filter(arr, feature_id, threshold, knn):

    """
    Filters a point cloud based on a given feature threshold. Only points
    with selected feature values higher than threshold are kept as valid.

    Parameters
    ----------
    arr : array
        Three-dimensional (m x n) array of a point cloud, where the
        coordinates are represented in the columns (n) and the points are
        represented in the rows (m).
    feature_id : int
        Column index of feature selected as criteria to filter. Column
        indices follow Python notation [0 - (n_columns - 1)].
    threshold : float
        Minimum feature value for valid points.
    knn : int
        Number of neighbors to select around each point. Used to describe
        local point arrangement.

    Returns
    -------
    mask_feature : numpy.ndarray
        Boolean mask with valid points entries set as True.

    """

    # Running NearestNeighborhood search and calculating geometric features
    # for each point's neighborhood.
    nbrs_idx = set_nbrs_knn(arr, arr, knn, False)
    features = knn_features(arr, nbrs_idx)
    # Masking valid points.
    return features[:, feature_id] >= threshold


def plane_filter(arr, rad, threshold):

    """
    Filters a point cloud based on its points planicity. Removes points that
    are part of a neighbohood with planar spatial arrangement (low curvature).

    Parameters
    ----------
    arr : array
        Three-dimensional (m x n) array of a point cloud, where the
        coordinates are represented in the columns (n) and the points are
        represented in the rows (m).
    rad : float
        Search radius distance around each point. Used to describe
        local point arrangement.
    threshold : float
        Minimum curvature value for valid points.

    Returns
    -------
    mask_plane : numpy.ndarray
        Boolean mask with valid points entries set as True.

    """

    # Running NearestNeighborhood search around each point in arr.
    nbrs_idx = set_nbrs_rad(arr, arr, rad, False)
    # Calculating curvature for each point's neighborhood.
    c = curvature(arr, nbrs_idx)

    return c >= threshold


def cluster_filter(arr, max_dist, eval_threshold):

    """
    Applies a cluster filter to a point cloud 'arr'. This filter aims to
    remove small, isolated, clusters of points.

    Parameters
    ----------
    arr : array
        Point cloud of shape n points x m dimensions to be filtered.
    max_dist : float
        Maximum distance between points to considered part of the same
        cluster.
    eval_threshold : float
        Minimum value for largest eigenvalue for a valid cluster. This value
        is an indication of cluster shape, in which the higher the eigenvalue,
        more elongated is the cluster. Points from clusters that have
        eigenvalue smaller then eval_threshold are filtered out.

    Returns
    -------
    mask : array
        Boolean mask of filtered points. Entries are set as True if belonging
        to a valid cluster and False otherwise.

    """

    # Initializing and fitting HDBSCAN clustering to input array 'arr'.
    clusterer = DBSCAN(max_dist).fit(arr)
    labels = clusterer.labels_

    # Initializing arrat of final eigenvalues for each cluster.
    final_evals = np.zeros([labels.shape[0], 3])
    # Looping over each unique cluster label.
    for L in np.unique(labels):
        # Obtaining indices for all entries in 'arr' that are part of current
        # cluster.
        ids = np.where(labels == L)[0]
        # Checking if current cluster is not an empty cluster (label == -1)
        # and if current cluster has more than 3 points.
        if (L != -1) & len(ids) >= 3:
            # Calculated eigenvalues for current cluster.
            e = svd_evals(arr[ids])
            # Assigning current eigenvalues to indices of all points of
            # current cluster in final_evals.
            final_evals[ids] = e

    # Calculate eigenvalues ratio. This standardizes all rows (eigenvalues
    # of each point) to an interval between 0 and 1. The sum of each row
    # is 1.
    ratio = np.asarray([i / np.sum(i) for i in final_evals])

    # Mask points by largest eigenvalue (column -0).
    return ratio[:, 0] >= eval_threshold


def radius_filter(arr, radius, min_points):

    """
    Applies a radius search filter, which remove isolated points/clusters of
    points.

    Parameters
    ----------
    arr : array
        Point cloud of shape n points x m dimensions to be filtered.
    radius : float
        Search radius around each point to form a neighborhood.
    min_point : int
        Minimum number of points in a neighborhood for it to be considered
        valid, i.e not filtered out.

    Returns
    -------
    mask : array
        Array of bools masking valid points as True and "noise" points as
        False.

    """

    # Setting up neighborhood indices.
    indices = set_nbrs_rad(arr, arr, radius, return_dist=False)

    # Allocating array of neighborhood's sizes (one entry for each point in
    # arr).
    n_points = np.zeros(arr.shape[0], dtype=int)

    # Iterating over each entry in indices and calculating total number of
    # points.
    for i, id_ in enumerate(indices):
        n_points[i] = id_.shape[0]

    return n_points >= min_points


def continuity_filter(wood, leaf, rad=0.05):

    """
    Function to apply a continuity filter to a point cloud that contains gaps
    defined as points from a second point cloud.
    This function works assuming that the continuous variable is the
    wood portion of a tree point cloud and the gaps in it are empty space
    or missclassified leaf data. In this sense, this function tries to correct
    gaps where leaf points are present.

    Parameters
    ----------
    wood : array
        Wood point cloud to be filtered.
    leaf : array
        Leaf point cloud, with points that may be causing discontinuities in
        the wood point cloud.
    rad : float
        Radius to search for neighboring points in the iterative process.

    Returns
    -------
    wood : array
        Filtered wood point cloud.
    not_wood : array
        Remaining point clouds after the filtering.

    """

    # Stacking wood and leaf arrays.
    arr = np.vstack((wood, leaf))

    # Getting root index (base_id) from point cloud 'arr'.
    base_id = np.argmin(arr[:, 2])

    # Calculating shortest path graph over sampled array.
    G = array_to_graph(arr, base_id, 3, 100, 0.05, 0.02, 0.5)
    node_ids, dist = extract_path_info(G, base_id, return_path=False)
    node_ids = np.array(node_ids)

    # Obtaining wood point cloud indices.
    wood_id = node_ids[node_ids <= wood.shape[0]]

    # Generating nearest neighbors search for the entire point cloud (arr).
    nbrs = NearestNeighbors(algorithm='kd_tree', leaf_size=10,
                            n_jobs=-1).fit(arr[node_ids])

    # Converting dist variable to array, as it is originaly a list.
    dist = np.asarray(dist)

    # Selecting points and accummulated distance for all wood points in arr.
    gp = arr[wood_id]
    d = dist[wood_id]

    # Preparing control variables to iterate over. idbase will be all initial
    # wood ids and pts all initial wood points. These variables are the ones
    # to use in search of possible missclassified neighbors.
    idbase = wood_id
    pts = gp

    # Setting treshold variables to iterative process.
    e = 9999999
    e_threshold = 3

    # Iterating until threshold is met.
    while e > e_threshold:

        # Obtaining the neighbor indices of current set of points (pts).
        idx2 = nbrs.radius_neighbors(pts, radius=rad,
                                     return_distance=False)

        # Initializing temporary variable id1.
        id1 = []
        # Looping over nn search indices and comparing their respective
        # distances to center point distance. If nearest neighbor distance (to
        # point cloud base) is smaller than center point distance, then ith
        # point is also wood.
        for i in range(idx2.shape[0]):
            for i_ in idx2[i]:
                if dist[i_] <= (d[i]):
                    id1.append(i_)

        # Uniquifying id1.
        id1 = np.unique(id1)

        # Comparing original idbase to new wood ids (id1).
        comp = np.in1d(id1, idbase)

        # Maintaining only new ids for next iteration.
        diff = id1[np.where(~comp)[0]]
        idbase = np.unique(np.hstack((idbase, id1)))

        # Passing new wood points to pts and recalculating e value.
        pts = arr[diff]
        e = pts.shape[0]

        # Passing accummulated distances from new points to d.
        d = dist[diff]

        # Stacking new points to initial wood points and removing duplicates.
        gp = np.vstack((gp, pts))
        gp = remove_duplicates(gp)

    # Removing duplicates from final wood points and obtaining not_wood points
    # from the difference between final wood points and full point cloud.
    wood = remove_duplicates(gp)
    not_wood = get_diff(wood, arr)

    return wood, not_wood


def array_majority(arr_1, arr_2, **kwargs):

    """
    Applies majority filter on two arrays.

    Parameters
    ----------
    arr_1 : array
        n-dimensional array of points to filter.
    arr_2 : array
        n-dimensional array of points to filter.
    **knn : int or float
        Number neighbors to select around each point in arr in order to apply
        the majority criteria.
    **rad : int or float
        Search radius arount each point in arr to select neighbors in order
        to apply the majority criteria.

    Returns
    -------
    c_maj_1 : array
        Boolean mask of filtered entries of same class as input 'arr_1'.
    c_maj_2 : array
        Boolean mask of filtered entries of same class as input 'arr_2'.

    Raises
    ------
    AssertionError:
        Raised if neither 'knn' or 'rad' arguments are passed with valid
        values (int or float).

    """

    # Asserting input arguments are valid.
    assert ('knn' in kwargs.keys()) or ('rad' in kwargs.keys()), 'Please\
 input a value for either "knn" or "rad".'

    if 'knn' in kwargs.keys():
        assert (type(kwargs['knn']) == int) or (type(kwargs['knn']) ==
                                                float), \
            '"knn" variable must be of type int or float.'
    elif 'rad' in kwargs.keys():
        assert (type(kwargs['rad']) == int) or (type(kwargs['rad']) ==
                                                float), \
            '"rad" variable must be of type int or float.'

    # Stacking the arrays from both classes to generate a combined array.
    arr = np.vstack((arr_1, arr_2))

    # Generating the indices for the local subsets of points around all points
    # in the combined array. Function used is based upon the argument passed.
    if 'knn' in kwargs.keys():
        indices = set_nbrs_knn(arr, arr, kwargs['knn'], return_dist=False)
    elif 'rad' in kwargs.keys():
        indices = set_nbrs_rad(arr, arr, kwargs['rad'], return_dist=False)

    # Making sure indices has type int.
    indices = indices.astype(int)

    # Generating the class arrays from both classified arrays and combining
    # them into a single classes array (classes).
    class_1 = np.full(arr_1.shape[0], 1, dtype=np.int)
    class_2 = np.full(arr_2.shape[0], 2, dtype=np.int)
    classes = np.hstack((class_1, class_2)).T

    # Allocating output variable.
    c_maj = np.zeros(classes.shape)

    # Selecting subset of classes based on the neighborhood expressed by
    # indices.
    class_ = classes[indices]

    # Looping over all points in indices.
    for i in range(len(indices)):

        # Counting the number of occurrences of each value in the ith instance
        # of class_.
        unique, count = np.unique(class_[i, :], return_counts=True)
        # Appending the majority class into the output variable.
        c_maj[i] = unique[np.argmax(count)]

    return c_maj == 1, c_maj == 2


def class_filter(arr_1, arr_2, target, **kwargs):

    """
    Function to apply class filter on an array based on the combination of
    classed from both arrays (arr_1 and arr_2). Which array gets filtered
    is defined by ''target''.

    Parameters
    ----------
    arr_1 : array
        n-dimensional array of points to filter.
    arr_2 : array
        n-dimensional array of points to filter.
    target : int or float
        Number of the input array to filter. Valid values are 0 or 1.
    **knn : int or float
        Number neighbors to select around each point in arr in order to apply
        the majority criteria.
    **rad : int or float
        Search radius arount each point in arr to select neighbors in order
        to apply the majority criteria.

    Returns
    -------
    c_maj_1 : array
        Boolean mask of filtered entries of same class as input 'arr_1'.
    c_maj_2 : array
        Boolean mask of filtered entries of same class as input 'arr_2'.

    Raises
    ------
    AssertionError:
        Raised if neither 'knn' or 'rad' arguments are passed with valid
        values (int or float).
    AssertionError:
        Raised if 'target' variable is not an int or float with value 0 or 1.

    """

    # Asserting input arguments are valid.
    assert ('knn' in kwargs.keys()) or ('rad' in kwargs.keys()), 'Please\
 input a value for either "knn" or "rad".'

    if 'knn' in kwargs.keys():
        assert (type(kwargs['knn']) == int) or (type(kwargs['knn']) ==
                                                float), \
            '"knn" variable must be of type int or float.'
    elif 'rad' in kwargs.keys():
        assert (type(kwargs['rad']) == int) or (type(kwargs['rad']) ==
                                                float), \
            '"rad" variable must be of type int or float.'

    assert (type(target) == int) or (type(target) == float), '"target"\
 variable must be of type int or float.'
    assert (target == 0) or (target == 1), '"target" variable must be either\
 0 or 1.'

    # Stacking the arrays from both classes to generate a combined array.
    arr = np.vstack((arr_1, arr_2))

    # Generating the class arrays from both classified arrays and combining
    # them into a single classes array (classes).
    class_1 = np.full(arr_1.shape[0], 0, dtype=np.int)
    class_2 = np.full(arr_2.shape[0], 1, dtype=np.int)
    classes = np.hstack((class_1, class_2)).T

    # Generating the indices for the local subsets of points around all points
    # in the combined array. Function used is based upon the argument passed.
    if 'knn' in kwargs.keys():
        indices = set_nbrs_knn(arr, arr, kwargs['knn'], return_dist=False)
    elif 'rad' in kwargs.keys():
        indices = set_nbrs_rad(arr, arr, kwargs['rad'], return_dist=False)

    # Making sure indices has type int.
    indices = indices.astype(int)

    # Allocating output variable.
    c_maj = classes.copy()

    # Selecting subset of classes based on the neighborhood expressed by
    # indices.
    class_ = classes[indices]

    # Checking for the target class.
    target_idx = np.where(classes == target)[0]

    # Looping over the target points to filter.
    for i in target_idx:

        # Counting the number of occurrences of each value in the ith instance
        # of class_.
        count = np.bincount(class_[i, :])
        # Appending the majority class into the output variable.
        c_maj[i] = count.argmax()

    return c_maj == 0, c_maj == 1


def dist_majority(arr_1, arr_2, **kwargs):

    """
    Applies majority filter on two arrays.

    Parameters
    ----------
    arr_1 : array
        n-dimensional array of points to filter.
    arr_2 : array
        n-dimensional array of points to filter.
    **knn : int or float
        Number neighbors to select around each point in arr in order to apply
        the majority criteria.
    **rad : int or float
        Search radius arount each point in arr to select neighbors in order to
        apply the majority criteria.

    Returns
    -------
    c_maj_1 : array
        Boolean mask of filtered entries of same class as input 'arr_1'.
    c_maj_2 : array
        Boolean mask of filtered entries of same class as input 'arr_2'.

    Raises:
    AssertionError:
        Raised if neither 'knn' or 'rad' arguments are passed with valid
        values (int or float).

    """

    # Asserting input arguments are valid.
    assert ('knn' in kwargs.keys()) or ('rad' in kwargs.keys()), 'Please\
 input a value for either "knn" or "rad".'

    if 'knn' in kwargs.keys():
        assert (type(kwargs['knn']) == int) or (type(kwargs['knn']) ==
                                                float), \
            '"knn" variable must be of type int or float.'
    elif 'rad' in kwargs.keys():
        assert (type(kwargs['rad']) == int) or (type(kwargs['rad']) ==
                                                float), \
            '"rad" variable must be of type int or float.'

    # Stacking the arrays from both classes to generate a combined array.
    arr = np.vstack((arr_1, arr_2))

    # Generating the indices for the local subsets of points around all points
    # in the combined array. Function used is based upon the argument passed.
    if 'knn' in kwargs.keys():
        dist, indices = set_nbrs_knn(arr, arr, kwargs['knn'])
    elif 'rad' in kwargs.keys():
        dist, indices = set_nbrs_rad(arr, arr, kwargs['rad'])

    # Making sure indices has type int.
    indices = indices.astype(int)

    # Generating the class arrays from both classified arrays and combining
    # them into a single classes array (classes).
    class_1 = np.full(arr_1.shape[0], 1, dtype=np.int)
    class_2 = np.full(arr_2.shape[0], 2, dtype=np.int)
    classes = np.hstack((class_1, class_2)).T

    # Allocating output variable.
    c_maj = np.zeros(classes.shape)

    # Selecting subset of classes based on the neighborhood expressed by
    # indices.
    class_ = classes[indices]

    # Looping over all points in indices.
    for i in range(len(indices)):

        # Obtaining classe from indices i.
        c = class_[i, :]
        # Caculating accummulated distance for each class.
        d1 = np.sum(dist[i][c == 1])
        d2 = np.sum(dist[i][c == 2])
        # Checking which class has the highest distance and assigning it
        # to current index in c_maj.
        if d1 >= d2:
            c_maj[i] = 1
        elif d1 < d2:
            c_maj[i] = 2

    return c_maj == 1, c_maj == 2
