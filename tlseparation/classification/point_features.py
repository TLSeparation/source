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


def curvature(arr, nbrs_idx):

    """
    Calculates pointwise curvature of a point cloud.

    Parameters
    ----------
    arr : array
        Three-dimensional (m x n) array of a point cloud, where the
        coordinates are represented in the columns (n) and the points are
        represented in the rows (m).
    nbr_idx : array
        N-dimensional array of indices from a nearest neighbors search of the
        point cloud in 'arr', where the rows (m) represents the points in
        'arr' and the columns represents the indices of the nearest neighbors
        from 'arr'.

    Returns
    -------
    c : numpy.ndarray
        1D (m x 1) array containing the curvature of each point in 'arr'.

    """

    # Allocating eigenvalues (evals) as array with shape n_points x 3 filled
    # with zeros.
    evals = np.zeros([arr.shape[0], 3], dtype=float)

    # Looping over each set of neighbors in nbrs_idx.
    for i, nids in enumerate(nbrs_idx):
        # Checking if local neighborhood of points contains more than 3
        # points. Otherwise, the calculation of eigenvalues/eigenvectors
        # is not possible.
        if arr[nids].shape[0] > 3:
            # Calculates ith eigenvalues using svd_evals.
            evals[i] = svd_evals(arr[nids])

    # Calculating curvature.
    c = evals[:, 2] / np.sum(evals, axis=1)

    return c


def knn_features(arr, nbr_idx, block_size=200000):

    """
    Calculates geometric descriptors: salient features and tensor features
    from an array and an indexing with fixed numbers of neighbors.

    Parameters
    ----------
    arr : array
        Three-dimensional (m x n) array of a point cloud, where the
        coordinates are represented in the columns (n) and the points are
        represented in the rows (m).
    nbr_idx : array
        N-dimensional array of indices from a nearest neighbors search of the
        point cloud in 'arr', where the rows (m) represents the points in
        'arr' and the columns represents the indices of the nearest neighbors
        from 'arr'.

    Returns
    -------
    features : array
        N-dimensional array (m x 6) of the calculated geometric descriptors.
        Where the rows (m) represent the points from 'arr' and the columns
        represents the features.

    """

    # Making sure block_size is limited by at most the number of points in
    # arr.
    if block_size > arr.shape[0]:
        block_size = arr.shape[0]

    # Creating block of ids.
    ids = np.arange(arr.shape[0])
    ids = np.array_split(ids, int(arr.shape[0] / block_size))

    # Making sure nbr_idx has the correct data type.
    nbr_idx = nbr_idx.astype(int)

    # Allocating s.
    s = np.zeros([arr.shape[0], 3], dtype=float)

    # Looping over blocks of ids to calculating eigenvalues for the
    # neighborhood around each point in arr.
    for i in ids:
        # Calculating the eigenvalues.
        s[i] = knn_evals(arr[nbr_idx[i]])

    # Calculating the ratio of the eigenvalues.
    ratio = (s.T / np.sum(s, axis=1)).T

    # Calculating the salient features and tensor features from the
    # eigenvalues ratio.
    features = calc_features(ratio)

    # Replacing the 'nan' values for 0.
    features[np.isnan(features)] = 0

    return features


def knn_evals(arr_stack):

    """
    Calculates eigenvalues of a stack of arrays.

    Parameters
    ----------
    arr_stack : array
        N-dimensional array (l x m x n) containing a stack of data, where the
        rows (m) represents the points coordinates, the columns (n) represents
        the axis coordinates and the layer (l) represents the stacks of points.

    Returns
    -------
    evals : array
        N-dimensional array (l x n) of eigenvalues calculated from
        'arr_stack'. The rows (l) represents the stack layers of points in
        'arr_stack' and the columns (n) represent the parameters in
        'arr_stack'.

    """

    # Calculating the covariance of the stack of arrays.
    cov = vectorized_app(arr_stack)

    # Calculating the eigenvalues using Singular Value Decomposition (svd).
    evals = np.linalg.svd(cov, compute_uv=False)

    return evals


def calc_features(e):

    """
    Calculates the geometric features using a set of eigenvalues, based on Ma
    et al. [#]_ and Wang et al. [#]_.

    Parameters
    ----------
    e : array
        N-dimensional array (m x 3) containing sets of 3 eigenvalues per
        row (m).

    Returns
    -------
    features : array
        N-dimensional array (m x 6) containing the calculated geometric
        features from 'e'.

    References
    ----------
    ..  [#] Ma et al., 2015. Improved Salient Feature-Based Approach for
            Automatically Separating Photosynthetic and Nonphotosynthetic
            Components Within Terrestrial Lidar Point Cloud Data of Forest
            Canopies.
    ..  [#] Wang et al., 2015. A Multiscale and Hierarchical Feature Extraction
            Method for Terrestrial Laser Scanning Point Cloud Classification.

    """

    # Calculating salient features.
    e1 = e[:, 2]
    e2 = e[:, 0] - e[:, 1]
    e3 = e[:, 1] - e[:, 2]

    # Calculating tensor features.
    t1 = (e[:, 1] - e[:, 2]) / e[:, 0]
    t2 = ((e[:, 0] * np.log(e[:, 0])) + (e[:, 1] * np.log(e[:, 1])) +
          (e[:, 2] * np.log(e[:, 2])))
    t3 = (e[:, 0] - e[:, 1]) / e[:, 0]

    return np.vstack(([e1, e2, e3, t1, t2, t3])).T


def vectorized_app(arr_stack):

    """
    Function to calculate the covariance of a stack of arrays. This function
    uses einstein summation to make the covariance calculation more efficient.
    Based on a reply from the user Divakar [#]_ at stackoverflow.

    Parameters
    ----------
    arr_stack : array
        N-dimensional array (l x m x n) containing a stack of data, where the
        rows (m) represents the points coordinates, the columns (n) represents
        the axis coordinates and the layer (l) represents the stacks of
        points.

    Returns
    -------
    cov : array
        N-dimensional array (l x n x n) of covariance values calculated from
        'arr_stack'. Each layer (l) contains a (n x n) covariance matrix
        calculated from the layers (l) in 'arr_stack'.

    References
    ----------
    ..  [#] Divakar, 2016. http://stackoverflow.com/questions/35756952/\
quickly-compute-eigenvectors-for-each-element-of-an-array-in-\
python.

    """

    # Centralizing the data around the mean.
    diffs = arr_stack - arr_stack.mean(1, keepdims=True)

    # Using the einstein summation of the centered data in regard to the array
    # stack shape to return the covariance of each array in the stack.
    return np.einsum('ijk,ijl->ikl', diffs, diffs)/arr_stack.shape[1]


def svd_evals(arr):

    """
    Calculates eigenvalues of an array using SVD.

    Parameters
    ----------
    arr : array
        nxm numpy.ndarray where n is the number of samples and m is the number
        of dimensions.

    Returns
    -------
    evals : array
        1xm numpy.ndarray containing the calculated eigenvalues in decrescent
        order.

    """

    # Calculating centroid coordinates of points in 'arr'.
    centroid = np.average(arr, axis=0)

    # Running SVD on centered points from 'arr'.
    _, evals, evecs = np.linalg.svd(arr - centroid, full_matrices=False)

    return evals
