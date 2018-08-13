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
__credits__ = ["Matheus Boni Vicari", "Phil Wilkes"]
__license__ = "GPL3"
__version__ = "1.3"
__maintainer__ = "Matheus Boni Vicari"
__email__ = "matheus.boni.vicari@gmail.com"
__status__ = "Development"


from collections import defaultdict


def voxelize_cloud(arr, voxel_size):

    """
    Generates a dictionary of voxels containing their central coordinates
    and indices of points belonging to each voxel.

    Parameters
    ----------
    arr: array
        Array of points/entries to voxelize.
    voxel_size: float
        Length of all voxels sides/edges.

    Returns
    -------
    vox: defaultdict
        Dictionary containing voxels. Keys are voxels' central coordinates and
        values are indices of points in arr inside each voxel.

    """

    voxels_ids = (arr / voxel_size).astype(int) * voxel_size
    vox = defaultdict(list)

    for i, v in enumerate(voxels_ids):
        vox[tuple(v)].append(i)

    return vox
