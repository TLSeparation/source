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
from knnsearch import (set_nbrs_knn, set_nbrs_rad)


def detect_rad_nn(arr, rad):

    """
    Calculates an average of number of neighbors based on a fixed radius
    around each point in a point cloud.

    Args:
        arr (array): Three-dimensional (m x n) array of a point cloud, where
            the coordinates are represented in the columns (n) and the points
            are represented in the rows (m).
        rad (float): Radius distance to select neighboring points.

    Returns
        mean_knn (int): Average number of points inside a radius 'rad' around
            each point in 'arr'.

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

    Args:
        arr (array): N-dimensional array (m x n) containing a set of
            parameters (n) over a set of observations (m).
        knn (int): Number of nearest neighbors to search to constitue the
            local subset of points around each point in 'arr'.

    Returns:
        dist (float): Optimal distance among neighboring points.

    """

    dist, indices = set_nbrs_knn(arr, arr, knn)

    return np.mean(dist[:, 1:]) + (np.std(dist[:, 1:]) * sigma)
