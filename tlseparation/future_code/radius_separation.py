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
from classification.point_features import calc_features
from numba import jit


@jit
def features_radius(arr, r):

    print('Initializing NearestNeighbors.\n')
    nbrs = NearestNeighbors(algorithm='kd_tree', n_jobs=-1).fit(arr)

    final_evals = np.zeros([arr.shape[0], 3])

    for i, p in enumerate(arr):
        idx = nbrs.radius_neighbors(p.reshape(1, -1), r,
                                    return_distance=False)[0]
        if len(idx) >= 3:
            final_evals[i, :] = evals(arr[idx])

    return final_evals


def features_radius_2(arr, r, block_size=300000):

    print('Initializing NearestNeighbors.\n')
    nbrs = NearestNeighbors(algorithm='kd_tree', n_jobs=-1).fit(arr)

    n_blocks = np.ceil(arr.shape[0] / block_size).astype(int)
    print('Data will be divided in %s blocks.\n' % n_blocks)

    pids = np.arange(arr.shape[0], dtype=int)
    blocks = block_array(pids, n_blocks)

    final_evals = np.zeros([arr.shape[0], 3])

    for bid, b in enumerate(blocks):
        print('Starting NearestNeighbors search on block %s.\n' % bid)

        block_arr = arr[b]

        idx = nbrs.radius_neighbors(block_arr, r, return_distance=False)

        for i, id_ in enumerate(idx):
            if len(id_) >= 3:
                final_evals[b[i], :] = evals(arr[id_])

    print('NearestNeighbors search finished.\n')

    return final_evals


def evals(arr):

    # Calculating centroid coordinates of points in 'arr'.
    centroid = np.average(arr, axis=0)

    # Running SVD on centered points from 'arr'.
    _, evals, evecs = np.linalg.svd(arr - centroid)

    return evals


def shannon_entropy(evals):

    if evals.ndim == 1:
        return ((evals[0] * np.log(evals[0])) - (evals[1] * np.log(evals[1])) -
                (evals[2] * np.log(evals[2])))

    elif evals.ndim == 2:
        return ((evals[:, 0] * np.log(evals[:, 0])) - (evals[:, 1] *
                np.log(evals[:, 1])) - (evals[:, 2] * np.log(evals[:, 2])))

    else:
        raise NameError('Wrond Dimensions')
        print('Input array must be 1D or 2D')


def block_array(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in xrange(n))