# Copyright (c) 2017-2019, Matheus Boni Vicari, TLSeparation Project
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
__copyright__ = "Copyright 2017-2019, TLSeparation Project"
__credits__ = ["Matheus Boni Vicari"]
__license__ = "GPL3"
__version__ = "1.3.2"
__maintainer__ = "Matheus Boni Vicari"
__email__ = "matheus.boni.vicari@gmail.com"
__status__ = "Development"

from ..utility import (cluster_features, cluster_size,
                       connected_component)


def isolated_clusters(arr, voxel_size=0.05, size_threshold=0.3,
                      feature_threshold=0.6, min_pts=10):
    
    """
    Performs a connected component analysis to cluster points from a point
    cloud and filters them these clusters based on size and shape (geometric
    feature).
    
    Parameters
    ----------
    arr : array
        Three-dimensional (m x n) array of a point cloud, where the
        coordinates are represented in the columns (n) and the points are
        represented in the rows (m).
    voxel_size: float
        Distance used to generate voxels from point cloud in order to 
        perform the connected component analysis in 3D space.
    size_threshold : int/float
        Minimum size, on any dimension, for a cluster to be set as
        valid (True)
    feature_threshold : float
        Minimum feature value for the cluster to be set as elongated (True).
    min_pts : int
        Minimum number of points for the cluster to be set as valid (True).
        
    Returns
    -------
    filter_mask : array
        1D mask array setting True for valid poins in 'arr' and False
        otherwise.
        
    """
    
    # Clustering points in 'arr' using connected_components.
    labels = connected_component(arr, voxel_size)
    # Filtering clustered points based on cluster size.
    filter_mask1 = cluster_size(arr, labels, size_threshold)
    # Filtering clustered points based on cluster geometric feature.
    filter_mask2 = cluster_features(arr, labels, feature_threshold,
                                     min_pts=min_pts)
    # Joining filter masks to generate the final mask.
    filter_mask = (filter_mask1 + filter_mask2).astype(bool)
    
    return arr[filter_mask], arr[~filter_mask]
    