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

import numpy as np
from sklearn.cluster import DBSCAN

def connected_component(arr, voxel_size):
    
    """
    Performs a connected component analysis to cluster points from a point
    cloud. 
    
    Parameters
    ----------
    arr : array
        Three-dimensional (m x n) array of a point cloud, where the
        coordinates are represented in the columns (n) and the points are
        represented in the rows (m).
    voxel_size: float
        Distance used to generate voxels from point cloud in order to 
        perform the connected component analysis in 3D space.
        
    Returns
    -------
    point_labels : array
        1D array with cluster labels assigned to each point from the input
        point cloud.
        
    """
    
    # Generate voxels central coordinates.
    voxel_coords = (arr / voxel_size).astype(int)
    # Initialize voxels and fills them based on the voxel coordinates for
    # each point.
    voxels = {}
    for i, v in enumerate(voxel_coords):
        if tuple(v) in voxels:
            voxels[tuple(v)].append(i)
        else:
            voxels[tuple(v)] = [i]
        
    # Running DBSCAN on the voxels created from the input point cloud. This
    # step takes advantage of the integer coordinates to cluster voxels
    # in a similar approach used in a classic connected components.
    db = DBSCAN(eps=1, min_samples=1, algorithm='kd_tree', metric='chebyshev',
                n_jobs=-1).fit(voxel_coords)
    labels = db.labels_
    
    # Assigning voxel cluster labels to each voxel's respective points.
    point_labels = np.full(arr.shape[0], -1, dtype=int)
    for l in np.unique(labels):
        mask = l == labels
        for c in voxel_coords[mask]:
            point_labels[voxels[tuple(c)]] = l
            
    return point_labels
