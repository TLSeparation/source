# Copyright (c) 2017, Matheus Boni Vicari, TLSeparation Project
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     1. Redistributions of source code must retain the above copyright notice,
#        this list of conditions and the following disclaimer.
#
#     2. Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#
#     3. Neither the name of the Raysect Project nor the names of its
#        contributors may be used to endorse or promote products derived from
#        this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

__author__ = "Matheus Boni Vicari"
__copyright__ = "Copyright 2017, TLSeparation Project"
__credits__ = ["Matheus Boni Vicari"]
__license__ = "GPL3"
__version__ = "1.1.5.1"
__maintainer__ = "Matheus Boni Vicari"
__email__ = "matheus.boni.vicari@gmail.com"
__status__ = "Development"

import numpy as np
from sklearn.neighbors import NearestNeighbors
from tlseparation.utility import *

knn = 100
voxel_size = 1
nbrs_threshold = voxel_size * 1.5
n_samples = 100

arr = np.loadtxt(r'D:\Dropbox\PhD\Data\LiDAR\ghana\2016\clouds/tree_5.txt')
point_cloud = remove_duplicates(arr[:, :3])

vox = voxelize_cloud(arr, voxel_size=voxel_size)
vox_coords = np.asarray(vox.keys())

base_id = np.argmin(vox_coords[:, 2])

G = array_to_graph(vox_coords, base_id, 3, knn, nbrs_threshold, 0.02)

accumulator = np.zeros([vox_coords.shape[0], n_samples], dtype=int)

for s in np.arange(n_samples):

    sid = np.random.choice(np.arange(vox_coords.shape[0]))

    nodes_ids, D, path_list = extract_path_info(G, sid, return_path=True)

    path_ids = [i for j in path_list.itervalues() for i in j]

    unique, count = np.unique(path_ids, return_counts=True)

    accumulator[unique, s] = count[unique]


