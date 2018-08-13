import numpy as np
from scipy.spatial.distance import cdist


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


def downsample_cloud(point_cloud, downsample_size, return_indices=False,
                     return_neighbors=False):

    """
    Downsamples a point cloud by voxelizing it and selecting points closest
    to the median coordinate of all points inside each voxel. The remaining
    points can be stored and returned as a dictrionary for later
    use in upsampling back to original input data.

    Parameters
    ----------
    point_cloud : numpy.ndarray
        Three-dimensional (m x n) array of a point cloud, where the
        coordinates are represented in the columns (n) and the points are
        represented in the rows (m).
    downsample_size : float
        Size of the voxels used to sample points into groups and select the
        most central point from. Note that this will not be the final points
        distance from each other, but an approximation.
    return_indices : bool
        Option to return results as downsampled array (False) or the
        indices of downsampled points from original point cloud (True).
    return_neighbors : bool
        Option to return original neighbors of downsampled points (True) or
        not (False). This information can be used to upsample back the
        downsampled indices.

    """

    # Voxelizing input point cloud by truncating coordinates based on
    # downsample_size.
    voxels_ids = (point_cloud / downsample_size).astype(int)
    voxels = {}

    # Looping over each point voxel index. Adds each point index to its
    # voxel key (vid).
    for i, vid in enumerate(voxels_ids):
        if tuple(vid) in voxels:
            voxels[tuple(vid)].append(i)
        else:
            voxels[tuple(vid)] = [i]

    # If return_neighbors is set to True, initialize neighbors_ids dictionary.
    if return_neighbors:
        neighbors_ids = {}

    # Initializing point cloud downsampled indices as array of zeros with
    # length equal to number of voxels.
    pc_downsample_ids = np.zeros(len(voxels.keys()), dtype=int)
    # Looping over each pair of voxel indices and point indices.
    for i, (vid, pids) in enumerate(voxels.iteritems()):
        # Calculating median coordinates of points inside current voxel.
        median_coord = np.median(point_cloud[pids], axis=0)
        # Calculating distance of every point inside current voxel to
        # their median.
        dist = cdist(point_cloud[pids], median_coord.reshape([1, 3]))
        # Sorting indices by distance and selecting closest point as
        # representative of current voxel's center. Assign selected point's
        # index to current index of pc_downsample_ids.
        sort_ids = np.argsort(dist.T)
        pids = np.array(pids).flatten()
        pc_downsample_ids[i] = pids[sort_ids[0][0]]
        # If set to return neighbors indices, assign all remaining points
        # indices to selected center index in neighbors_ids.
        if return_neighbors:
            neighbors_ids[pc_downsample_ids[i]] = pids[sort_ids[0]]

    if return_indices:
        if return_neighbors:
            return pc_downsample_ids, neighbors_ids
        else:
            return pc_downsample_ids
    else:
        if return_neighbors:
            return point_cloud[pc_downsample_ids], neighbors_ids
        else:
            return point_cloud[pc_downsample_ids]


def upsample_cloud(upsample_ids, neighbors_dict):

    """
    Upsample cloud based on downsampling information from 'downsample_cloud'.
    This function will loop over each 'upsample_ids' and retrieve its
    original neighboring points stored in 'neighbors_dict'.

    Parameters
    ----------
    upsample_ids : list
        List of indices in 'neighbors_dict' to upsample.
    neighbors_dict : dict
        Neighbors information provided by 'downsample_cloud' containing
        all the original neighboring points to each point in the downsampled
        cloud.

    Returns
    -------
    upsampled_indices : numpy.ndarray
        Upsampled points from original point cloud.

    """

    # Looping over each index in upsample_ids and retrieving its
    # original neighbors indices.
    ids = [neighbors_dict[i] for i in upsample_ids if i in neighbors_dict]

    return np.unique([i for j in ids for i in j])
