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
__credits__ = ["Matheus Boni Vicari", "Phil Wilkes"]
__license__ = "GPL3"
__version__ = "1.2.1.5"
__maintainer__ = "Matheus Boni Vicari"
__email__ = "matheus.boni.vicari@gmail.com"
__status__ = "Development"

import numpy as np
import datetime
from ..classification import (wlseparate_abs, wlseparate_ref_voting,
                              detect_main_pathways, get_base, DefaultClass)
from ..utility import (get_diff, remove_duplicates, radius_filter,
                       class_filter, cluster_filter, continuity_filter,
                       detect_nn_dist, voxelize_cloud)


def large_tree_1(arr, class_file=[], cont_filt=True,
                 class_prob_threshold=0.95, verbose=False):

    """
    Run an automated separation of a single tree point cloud.

    Parameters
    ----------
    arr : array
        Three-dimensional point cloud of a single tree to perform the
        wood-leaf separation. This should be a n-dimensional array (m x n)
        containing a set of coordinates (n) over a set of points (m).
    class_file : str
        Path to classes reference values file. This file will be loaded and
        its reference values are used to select wood and leaf classes.
    cont_filt : boolean
        Option to select if continuity_filter should be applied to wood and
        leaf point clouds. Default is True.
    class_prob_threshold : float
        Classification probability threshold to filter classes. This aims to
        avoid selecting points that are not confidently enough assigned to
        any given class. Default is 0.95.

    Returns
    -------
    wood_final : array
        Wood point cloud.
    leaf_final : array
        Leaf point cloud.

    """

    # Checking input class_file, if it's an empty list, use default values.
    if len(class_file) == 0:
        class_file = DefaultClass().ref_table

    ###########################################################################
    # Making sure input array has only 3 dimensions and no duplicated points.
    if verbose:
        print(str(datetime.datetime.now()) + ' | removing duplicates')
    arr = remove_duplicates(arr[:, :3])

    # Calculating recommended distance between neighboring points.
    if verbose:
        print(str(datetime.datetime.now()) + ' | calculating recommended \
distance between neighboring points')
    nndist = detect_nn_dist(arr, 10, 0.5)
    if verbose:
        print(str(datetime.datetime.now()) + ' | nndist: %s' % nndist)

    ###########################################################################
    # Obtaining mask of points from a slice of points located at the base of
    # the tree.
    if verbose:
        print(str(datetime.datetime.now()) + ' | obtaining mask of points \
from a slice of points located at the base of the tree')

    try:
        base_mask = get_base(arr, 0.5)
        base_ids = np.where(base_mask)[0]
    except:
        base_ids = []
        print('Failed to obtain base_mask.')

    # Masking points most likely to be part of the trunk and larger branches.
    if verbose:
        print(str(datetime.datetime.now()) + ' | masking points most likely \
to be part of the trunk and larger branches')
    try:
        # Voxelizing cloud:
        vox = voxelize_cloud(arr, voxel_size=.1)
        vox_coords = np.asarray(vox.keys())

        # Using detect_main_pathways on voxels coordinates.
        trunk_mask_voxel = detect_main_pathways(vox_coords, 80, 100, .15,
                                                verbose=verbose)
        # Obtaining indices of point in arr that are inside voxels masked as
        # trunk (True) in trunk_mask_voxel.
        trunk_ids = np.unique([j for i in vox_coords[trunk_mask_voxel] for
                               j in vox[tuple(i)]])

        # Setting up trunk mask, as its opposite represents points not
        # detected as trunk.
        trunk_mask = np.zeros(arr.shape[0], dtype=bool)
        trunk_mask[trunk_ids] = True

        # Obtaining indices of points that were not detected as part of the
        # trunk.
        not_trunk_ids = np.where(~trunk_mask)[0].astype(int)
    except:
        trunk_ids = []
        print('Failed to obtain trunk_mask.')

    ###########################################################################
    try:
        if verbose:
            print(str(datetime.datetime.now()) + ' | performing absolute \
threshold separation on points not detected as trunk (not_trunk_ids)')
        # Performing absolute threshold separation on points not detected
        # as trunk (not_trunk_ids).
        ids_1, prob_1 = wlseparate_abs(arr[not_trunk_ids], 40,
                                       n_classes=4)

        # Obtaining wood_1 ids and classification probability.
        if verbose:
            print(str(datetime.datetime.now()) + ' | obtaining wood_1 ids \
and classification probability')
        wood_1_mask = not_trunk_ids[ids_1['wood']]
        wood_1_prob = prob_1['wood']
        # Filtering out points that were classified with a probability lower
        # than class_prob_threshold.
        if verbose:
            print(str(datetime.datetime.now()) + ' | filtering out points \
that were classified with a probability lower than class_prob_threshold')
        wood_1 = wood_1_mask[wood_1_prob >= class_prob_threshold]

    except:
        # In case absolute threshold separation fails, set wood_1_2 as an
        # empty list.
        wood_1 = []
        if verbose:
            print(str(datetime.datetime.now()) + ' | absolute threshold \
separation failed, setting wood_1_2 as an empty list')
    ###########################################################################
    try:
        # Performing reference class voting separation on the whole input point
        # cloud.
        # Defining list of knn values to use in the voting scheme.
        knn_list = [80, 70, 100, 150]
        # Running reference class voting separation.
        if verbose:
            print(str(datetime.datetime.now()) + ' | running reference class \
voting separation')
        ids_2, count_2, prob_2 = wlseparate_ref_voting(arr, knn_list,
                                                       class_file, n_classes=4)

        # Obtaining indices and classification probabilities for classes
        # twig and trunk (both components of wood points).
        twig_2_mask = ids_2['twig']
        twig_2_prob = prob_2['twig']
        trunk_2_mask = ids_2['trunk']
        trunk_2_prob = prob_2['trunk']

        # Masking twig and trunk classes by classification probability
        # threshold.
        twig_2_prob_mask = twig_2_prob >= class_prob_threshold
        trunk_2_prob_mask = trunk_2_prob >= class_prob_threshold

        # Obtaining twig_2 and trunk_2 vote counts, which are the number of
        # votes that each point in twig_2 and trunk_2 received to be
        # classified as such.
        twig_2_count = count_2['twig']
        trunk_2_count = count_2['trunk']

        # Filtering twig_2 and trunk_2 by a minimun number of votes. Point
        # indices with number of votes smaller than the defined threshold
        # are left out.
        twig_2 = twig_2_mask[twig_2_count >= 2][twig_2_prob_mask]
        trunk_2 = trunk_2_mask[trunk_2_count >= 2][trunk_2_prob_mask]

    except:
        # In case voting separation fails, set twig_2_2 and trunk_2_2 as
        # empty lists.
        twig_2 = []
        trunk_2 = []
        raise

    ###########################################################################
    # Stacking all clouds part of the wood portion.
    wood_ids = np.hstack((base_ids, trunk_ids, twig_2, trunk_2, wood_1))
    wood_ids = np.unique(wood_ids).astype(int)

    wood_2_mask = radius_filter(arr[wood_ids], 0.1, 5)
    wood_2_ids = wood_ids[wood_2_mask]


    # Removing duplicate points from wood point cloud. As there is a lot of
    # overlap in the classification phase, this step is rather necessary.
    if verbose:
        print(str(datetime.datetime.now()) + ' | removing duplicate points \
from wood point cloud. As there is a lot of overlap in the classification \
phase, this step is rather necessary')
    wood = remove_duplicates(wood)
    # Obtaining leaf point cloud from the difference between input cloud 'arr'
    # and wood points.
    if verbose:
        print(str(datetime.datetime.now()) + ' | obtaining leaf point cloud \
from the difference between input cloud "arr" and wood points')
    leaf = get_diff(arr, wood)

    ###########################################################################
    if cont_filt:
        # Applying continuity filter in an attempt to close gaps in the wood
        # point cloud (i.e. misclassified leaf points in between portions of
        # wood points).
        if verbose:
            print(str(datetime.datetime.now()) + ' | applying continuity \
filter in an attempt to close gaps in the wood point cloud')
        try:
            wood_final, leaf_final = continuity_filter(wood, leaf, rad=nndist)
        except:
            wood_final = wood
            leaf_final = leaf
    else:
        wood_final = wood
        leaf_final = leaf

    ###########################################################################

    return wood_final, leaf_final