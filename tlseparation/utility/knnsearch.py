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
from sklearn.neighbors import NearestNeighbors


def set_nbrs_knn(arr, pts, knn, return_dist=True, block_size=100000):

    """
    Function to create a set of nearest neighbors indices and their respective
    distances for a set of points. This function uses a knn search and sets a
    limit size for a block of points to query. This makes it less efficient in
    terms of processing time, but avoids running out of memory in cases of
    very dense/large arrays/queries.

    Parameters
    ----------
    arr : array
        N-dimensional array to perform the knn search on.
    pts : array
        N-dimensional array to search for on the knn search.
    knn : int
        Number of nearest neighbors to search for.
    return_dist : boolean
        Option to return or not the distances of each neighbor.
    block_size : int
        Limit of points to query. The variable 'pts' will be subdivided in n
        blocks of size block_size to perform query.

    Returns
    -------
    indices : array
        Set of neighbors indices from 'arr' for each entry in 'pts'.
    distance : array
        Distances from each neighbor to each central point in 'pts'.

    """

    # Making sure knn is of type int.
    knn = int(knn)

    # Initiating the nearest neighbors search and fitting it to the input
    # array.
    nbrs = NearestNeighbors(n_neighbors=knn, metric='euclidean',
                            algorithm='kd_tree', leaf_size=15,
                            n_jobs=-1).fit(arr)

    # Making sure block_size is limited by at most the number of points in
    # arr.
    if block_size > pts.shape[0]:
        block_size = pts.shape[0]

    # Creating block of ids.
    ids = np.arange(pts.shape[0])
    ids = np.array_split(ids, int(pts.shape[0] / block_size))

    # Initializing variables to store distance and indices.
    if return_dist is True:
        distance = np.zeros([pts.shape[0], knn])
    indices = np.zeros([pts.shape[0], knn])

    # Checking if the function should return the distance as well or only the
    # neighborhood indices.
    if return_dist is True:
        # Obtaining the neighborhood indices and their respective distances
        # from the center point by looping over blocks of ids.
        for i in ids:
            nbrs_dist, nbrs_ids = nbrs.kneighbors(pts[i])
            distance[i] = nbrs_dist
            indices[i] = nbrs_ids
        return distance, indices

    elif return_dist is False:
        # Obtaining the neighborhood indices only  by looping over blocks of
        # ids.
        for i in ids:
            nbrs_ids = nbrs.kneighbors(pts[i], return_distance=False)
            indices[i] = nbrs_ids
        return indices


def set_nbrs_rad(arr, pts, rad, return_dist=True, block_size=100000):

    """
    Function to create a set of nearest neighbors indices and their respective
    distances for a set of points. This function uses a radius search and sets
    a limit size for a block of points to query. This makes it less efficient
    in terms of processing time, but avoids running out of memory in cases of
    very dense/large arrays/queries.

    Parameters
    ----------
    arr : array
        N-dimensional array to perform the radius search on.
    pts : array
        N-dimensional array to search for on the knn search.
    rad : float
        Radius of the NearestNeighbors search.
    return_dist : boolean
        Option to return or not the distances of each neighbor.
    block_size : int
        Limit of points to query. The variable 'pts' will be subdivided in n
        blocks of size block_size to perform query.

    Returns
    -------
    indices : array
        Set of neighbors indices from 'arr' for each entry in 'pts'.
    distance : array
        Distances from each neighbor to each central point in 'pts'.

    """

    # Making sure block_size is limited by at most the number of points in
    # arr.
    if block_size > pts.shape[0]:
        block_size = pts.shape[0]

    # Initiating the nearest neighbors search and fitting it to the input
    # array.
    nbrs = NearestNeighbors(radius=rad, metric='euclidean',
                            algorithm='kd_tree', leaf_size=15,
                            n_jobs=-1).fit(arr)

    # Creating block of ids.
    ids = np.arange(pts.shape[0])
    ids = np.array_split(ids, int(pts.shape[0] / block_size))

    # Initializing variables to store distance and indices.
    if return_dist is True:
        distance = []
    indices = []

    # Checking if the function should return the distance as well or only the
    # neighborhood indices.
    if return_dist is True:
        # Obtaining the neighborhood indices and their respective distances
        # from the center point by looping over blocks of ids.
        for i in ids:
            nbrs_dist, nbrs_ids = nbrs.radius_neighbors(pts[i])
            for j, k in enumerate(i):
                distance.append(nbrs_dist[j])
                indices.append(nbrs_ids[j])
        return distance, indices

    elif return_dist is False:
        # Obtaining the neighborhood indices only  by looping over blocks of
        # ids.
        for i in ids:
            nbrs_ids = nbrs.radius_neighbors(pts[i], return_distance=False)
            for j, k in enumerate(i):
                indices.append(nbrs_ids[j])
        return indices


def subset_nbrs(distance, indices, new_knn, block_size=100000):

    """
    Performs a subseting of points from the results of a nearest neighbors
    search.
    This function assumes that the first index/distance in each row represents
    the center point of the neighborhood represented by said rows.

    Parameters
    ----------
    distance : array
        Distances from each neighbor to each central point in 'pts'.
    indices : array
        Set of neighbors indices from 'arr' for each entry in 'pts'.
    new_knn : array
        Number of neighbors to select from the initial number of neighbors.
    block_size : int
        Limit of points to query. The variables 'distance' and 'indices' will
        be subdivided in n blocks of size block_size to perform query.

    Returns
    -------
    distance : array
        Subset of distances from each neighbor 'indices'.
    indices : array
        Subset of neighbors indices from 'indices'.

    """

    # Making sure block_size is limited by at most the number of points in
    # arr.
    if block_size > distance.shape[0]:
        block_size = distance.shape[0]

    # Creating block of ids.
    ids = np.arange(distance.shape[0])
    ids = np.array_split(ids, int(distance.shape[0] / block_size))

    # Initializing new_distance and new_indices variables.
    new_distance = []
    new_indices = []

    # Processing all blocks of indices in ids.
    for id_ in ids:

        # Looping over each sample in distance and indices.
        for d, i in zip(distance[id_], indices[id_]):
            # Checks if new knn values are smaller than current distance and
            # indices rows. This avoids errors of trying to select a number of
            # columns larger than the available columns.
            if distance.shape[1] >= new_knn:
                new_distance.append(d[:new_knn+1])
                new_indices.append(i[:new_knn+1].astype(int))
            else:
                new_distance.append(d)
                new_indices.append(int(i))

    # Returning new_distance and new_indices as arrays.
    return np.asarray(new_distance), np.asarray(new_indices)
