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
from sklearn.cluster import DBSCAN


def cluster_filter(arr, max_dist, min_points):

    clusterer = DBSCAN(eps=max_dist, n_jobs=-1).fit(arr)
    labels = clusterer.labels_

    final_evals = np.zeros([labels.shape[0], 3])
    for l in np.unique(labels):
        ids = np.where(labels == l)[0]
        if l != -1:
            if ids.shape[0] >= min_points:
                e = evals(arr[ids])
                final_evals[ids] = e

#    MELHORAR MASK (USAR FEATURES?)
    return final_evals[:, 0] > 10

def calc_features(e):

    """
    Calculates the geometric features using a set of eigenvalues, based on Ma
    et al. (2015) and Wang et al. (2015).

    Args:
        e (array): N-dimensional array (m x 3) containing sets of 3
            eigenvalues per row (m).

    Returns:
        features (array): N-dimensional array (m x 6) containing the
            calculated geometric features from 'e'.

    Reference:
    ..  [1] Ma et al., 2015. Improved Salient Feature-Based Approach for
            Automatically Separating Photosynthetic and Nonphotosynthetic
            Components Within Terrestrial Lidar Point Cloud Data of Forest
            Canopies.
    ..  [2] Wang et al., 2015. A Multiscale and Hierarchical Feature Extraction
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

def evals(arr):

    # Calculating centroid coordinates of points in 'arr'.
    centroid = np.average(arr, axis=0)

    # Running SVD on centered points from 'arr'.
    _, evals, evecs = np.linalg.svd(arr - centroid)

    return evals