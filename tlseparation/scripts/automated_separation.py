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
import sys
sys.path.append('..')
from classification import (wlseparate_abs, wlseparate_ref_voting,
                            detect_main_pathways, get_base)
from utility import (get_diff, remove_duplicates, radius_filter, class_filter,
                     cluster_filter, continuity_filter, detect_nn_dist)


def large_tree_1(arr, class_file, cont_filt=True, class_prob_threshold=0.95):

    """
    Run an automated separation of a single tree point cloud.

    Args:
        arr (array): Three-dimensional point cloud of a single tree to
            perform the wood-leaf separation. This should be a n-dimensional
            array (m x n) containing a set of coordinates (n) over a set of
            points (m).
            class_file: str
        class_file (str): Path to classes reference values file. This file will
            be loaded and its reference values are used to select wood and
            leaf classes.
        cont_filt (boolean): Option to select if continuity_filter should
            be applied to wood and leaf point clouds. Default is True.
        class_prob_threshold (float): Classification probability threshold
            to filter classes. This aims to avoid selecting points that are
            not confidently enough assigned to any given class. Default is
            0.95.

    Returns:
        wood_final (array): Wood point cloud.
        leaf_final (array): Leaf point cloud.

    """

    ###########################################################################
    # Making sure input array has only 3 dimensions and no duplicated points.
    arr = remove_duplicates(arr[:, :3])

    # Calculating recommended distance between neighboring points.
    nndist = detect_nn_dist(arr, 10, 0.5)

    ###########################################################################
    # Obtaining mask of points from a slice of points located at the base of
    # the tree.
    try:
        base_mask = get_base(arr, 0.5)
        base_ids = np.where(base_mask)[0]
    except:
        base_ids = []
        print('Failed to obtain base_mask.')

    # Masking points most likely to be part of the trunk and larger branches.
    try:
        trunk_mask = detect_main_pathways(arr, 80, 100, nndist)
        trunk_ids = np.where(trunk_mask)[0]
        not_trunk_ids = np.where(~trunk_mask)[0].astype(int)
    except:
        trunk_ids = []
        print('Failed to obtain trunk_mask.')

    ###########################################################################
    try:
        # Performing absolute threshold separation on points not detected
        # as trunk (not_trunk_ids).
        ids_1, prob_1 = wlseparate_abs(arr[not_trunk_ids], 40,
                                       n_classes=4)

        # Obtaining wood_1 ids and classification probability.
        wood_1_mask = not_trunk_ids[ids_1['wood']]
        wood_1_prob = prob_1['wood']
        # Filtering out points that were classified with a probability lower
        # than class_prob_threshold.
        wood_1 = wood_1_mask[wood_1_prob >= class_prob_threshold]

        try:
            # Applying class_filter to remove wood_1 points that are more
            # likely to be part of a leaf point cloud (not_wood_1).
            wood_1_1_mask, _ = class_filter(arr[wood_1], arr[~wood_1], 1,
                                            knn=20)
            # Obtaining wood_1 filtered point indices.
            wood_1_1_mask = np.where(wood_1_1_mask)[0]
            wood_1_1 = wood_1[wood_1_1_mask]
        except:
            wood_1_1 = wood_1

        try:
            # Applying cluster_filter to remove isolated clusters of points in
            # wood_1_1.
            wood_1_2_mask = cluster_filter(arr[wood_1_1], 0.1, 5, 1.5)
            # Obtaining indices for final wood_1_* class.
            wood_1_2 = wood_1_1[wood_1_2_mask]
        except:
            wood_1_2 = wood_1_1

    except:
        # In case absolute threshold separation fails, set wood_1_2 as an
        # empty list.
        wood_1_2 = []

    ###########################################################################
    try:
        # Performing reference class voting separation on the whole input point
        # cloud.
        # Defining list of knn values to use in the voting scheme.
        knn_list = [80, 70, 100, 150]
        # Running reference class voting separation.
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

        try:
            # Applying radius_filter on filtered twig point cloud.
            twig_2_1_mask = radius_filter(arr[twig_2], 0.05, 10)
            twig_2_1 = twig_2[twig_2_1_mask]
        except:
            twig_2_1 = twig_2

        try:
            # Applying cluster_filter to remove isolated clusters of points in
            # twig_2_mask.
            twig_2_2_mask = cluster_filter(arr[twig_2_1], 0.1, 20, 3)
            twig_2_2 = twig_2_1[twig_2_2_mask]
        except:
            twig_2_2 = twig_2_1

        try:
            # Applying radius_filter to trunk_2 point cloud.
            trunk_2_1_mask = radius_filter(arr[trunk_2], 0.05, 10)
            trunk_2_1 = trunk_2[trunk_2_1_mask]
        except:
            trunk_2_1 = trunk_2

        try:
            # Applying cluster_filter to remove isolated clusters of points in
            # twig_2_mask.
            trunk_2_2_mask = cluster_filter(arr[trunk_2_1], 0.1, 20, 3)
            trunk_2_2 = trunk_2_1[trunk_2_2_mask]
        except:
            trunk_2_2 = trunk_2_1

    except:
        # In case voting separation fails, set twig_2_2 and trunk_2_2 as
        # empty lists.
        twig_2_2 = []
        trunk_2_2 = []
        raise

    ###########################################################################
    # Stacking all clouds part of the wood portion.
    wood_ids = np.hstack((base_ids, trunk_ids, twig_2_2, trunk_2_2, wood_1_2))
    wood_ids = np.unique(wood_ids).astype(int)
    wood = arr[wood_ids]
    # Removing duplicate points from wood point cloud. As there is a lot of
    # overlap in the classification phase, this step is rather necessary.
    wood = remove_duplicates(wood)
    # Obtaining leaf point cloud from the difference between input cloud 'arr'
    # and wood points.
    leaf = get_diff(arr, wood)

    ###########################################################################
    if cont_filt:
        # Applying continuity filter in an attempt to close gaps in the wood
        # point cloud (i.e. misclassified leaf points in between portions of
        # wood points).
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
