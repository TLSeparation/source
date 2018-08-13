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
from peakdetect import peakdet


def detect_optimal_knn(arr, rad_lst=[0.1, 0.2, 0.3], sample_size=10000):

    """
    Detects optimal values for knn in order to facilitate material separation.

    Parameters
    ----------
    arr: array
        Set of 3D points.
    rad_lst: list
        Set of radius values to generate samples of neighborhoods. This is
        used to select points to calculate a number of neighboring points
        distribution from the point cloud.
    sample_size: int
        Number of points in arr to process in order to genrate a distribution.

    Returns
    -------
    knn_lst: list
        Set of k-nearest neighbors values.

    """

    # Generating sample indices.
    sids = np.random.choice(np.arange(arr.shape[0]), sample_size,
                            replace=False)

    # Obtaining nearest neighbors' indices and distance for sampled points.
    # This process is done just once, with the largest value of radius in
    # rad_lst. Later on, it is possible to subsample indices by limiting
    # their distances for a smaller radius.
    dist, ids = set_nbrs_rad(arr, arr[sids], np.max(rad_lst), True)

    # Initializing empty list to store knn values.
    knn_lst = []

    # Looping over each radius value.
    for r in rad_lst:
        # Counting number of points inside radius r.
        n_pts = [len(i[d <= r]) for i, d in zip(ids, dist)]

        # Binning n_pts into a histogram.
        y, x = np.histogram(n_pts)

        # Detecting peaks of accumulated points from n_pts.
        maxtab, mintab = peakdet(y, 100)
        maxtab = np.array(maxtab)

        # Appending knn values relative to peaks detected in n_pts.
        knn_lst.append(x[maxtab[:, 0]])

    # Flattening nested lists into a final list of knn values.
    knn_lst = [i for j in knn_lst for i in j]

    return knn_lst


def detect_rad_nn(arr, rad):

    """
    Calculates an average of number of neighbors based on a fixed radius
    around each point in a point cloud.

    Parameters
    ----------
    arr : array
        Three-dimensional (m x n) array of a point cloud, where the
        coordinates are represented in the columns (n) and the points are
        represented in the rows (m).
    rad : float
        Radius distance to select neighboring points.

    Returns
    -------
    mean_knn : int
        Average number of points inside a radius 'rad' around each point in
        'arr'.

    """

    # Performin Nearest Neighbors search for the whole point cloud.
    indices = set_nbrs_rad(arr, arr, rad, return_dist=False)

    # Counting number of points around each point in 'arr'.
    indices_len = [len(i) for i in indices]

    # Calculates a mean of all neighboring point counts.
    mean_knn = np.mean(indices_len).astype(int)

    return mean_knn


def detect_nn_dist(arr, knn, sigma=1):

    """
    Calcuates the optimum distance among neighboring points.

    Parameters
    ----------
    arr : array
        N-dimensional array (m x n) containing a set of parameters (n) over
        a set of observations (m).
    knn : int
        Number of nearest neighbors to search to constitue the local subset
        of points around each point in 'arr'.

    Returns
    -------
    dist : float
        Optimal distance among neighboring points.

    """

    dist, indices = set_nbrs_knn(arr, arr, knn)

    return np.mean(dist[:, 1:]) + (np.std(dist[:, 1:]) * sigma)
