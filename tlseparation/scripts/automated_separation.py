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
__version__ = "1.2.2.1"
__maintainer__ = "Matheus Boni Vicari"
__email__ = "matheus.boni.vicari@gmail.com"
__status__ = "Development"

import numpy as np
import datetime
from ..classification import (wlseparate_abs, wlseparate_ref_voting,
                              detect_main_pathways, voxel_path_detection,
                              get_base, DefaultClass)
from ..utility import (get_diff, remove_duplicates, radius_filter,
                       class_filter, cluster_filter, continuity_filter,
                       detect_nn_dist, voxelize_cloud, detect_optimal_knn)


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

    # Calculating recommended distance between neighboring points and optimal
    # knn values.
    if verbose:
        print(str(datetime.datetime.now()) + ' | calculating recommended \
distance between neighboring points')
    nndist = detect_nn_dist(arr, 10, 0.5)
    knn_lst = detect_optimal_knn(arr)
    knn = np.min(knn_lst)
    knn_lst = [80, 70, 100, 150]

    if verbose:
        print(str(datetime.datetime.now()) + ' | nndist: %s' % nndist)
        print(str(datetime.datetime.now()) + ' | knn_lst: %s' % knn_lst)

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
        ids_1, prob_1 = wlseparate_abs(arr[not_trunk_ids], knn,
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

        try:
            # Applying class_filter to remove wood_1 points that are more
            # likely to be part of a leaf point cloud (not_wood_1).
            if verbose:
                print(str(datetime.datetime.now()) + ' | \
applying class_filter to remove wood_1 points that are more likely to be \
part of a leaf point cloud (not_wood_1)')
            # Setting up a boolean mask of wood_1 and not_wood_1 points.
            wood_1_bool = np.zeros(arr.shape[0], dtype=bool)
            wood_1_bool[wood_1] = True
            wood_1_1_mask, _ = class_filter(arr[wood_1_bool],
                                            arr[~wood_1_bool], 0, knn=10)
            # Obtaining wood_1 filtered point indices.
            if verbose:
                print(str(datetime.datetime.now()) + ' | obtaining wood_1 \
filtered point indices')
            wood_1_1_mask = np.where(wood_1_1_mask)[0]
            wood_1_1 = wood_1[wood_1_1_mask]
        except:
            wood_1_1 = wood_1

        try:
            # Applying cluster_filter to remove isolated clusters of points in
            # wood_1_1.
            wood_1_2_mask = cluster_filter(arr[wood_1_1], 10, 0.3)
            if verbose:
                print(str(datetime.datetime.now()) + ' | applying \
cluster_filter to remove isolated clusters of points in wood_1_1')
            # Obtaining indices for final wood_1_* class.
            wood_1_2 = wood_1_1[wood_1_2_mask]
        except:
            wood_1_2 = wood_1_1

    except:
        # In case absolute threshold separation fails, set wood_1_2 as an
        # empty list.
        wood_1_2 = []
        if verbose:
            print(str(datetime.datetime.now()) + ' | absolute threshold \
separation failed, setting wood_1_2 as an empty list')
    ###########################################################################
    try:
        # Performing reference class voting separation on the whole input point
        # cloud.
        # Running reference class voting separation.
        if verbose:
            print(str(datetime.datetime.now()) + ' | running reference class \
voting separation')
        ids_2, count_2, prob_2 = wlseparate_ref_voting(arr, knn_lst,
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
            if verbose:
                print(str(datetime.datetime.now()) + ' | applying \
radius_filter on filtered twig point cloud')
            twig_2_1_mask = radius_filter(arr[twig_2], 0.1, 5)
            twig_2_1 = twig_2[twig_2_1_mask]
        except:
            twig_2_1 = twig_2

        try:
            # Applying cluster_filter to remove isolated clusters of points in
            # twig_2_mask.
            if verbose:
                print(str(datetime.datetime.now()) + ' | applying \
cluster_filter to remove isolated clusters of points in twig_2_mask')
            twig_2_2_mask = cluster_filter(arr[twig_2_1], 10, 0.3)
            twig_2_2 = twig_2_1[twig_2_2_mask]
        except:
            twig_2_2 = twig_2_1

        try:
            # Applying radius_filter to trunk_2 point cloud.
            if verbose:
                print(str(datetime.datetime.now()) + ' | applying \
radius_filter to trunk_2 point cloud')
            trunk_2_1_mask = radius_filter(arr[trunk_2], 0.1, 5)
            trunk_2_1 = trunk_2[trunk_2_1_mask]
        except:
            trunk_2_1 = trunk_2

        try:
            # Applying cluster_filter to remove isolated clusters of points in
            # twig_2_mask.
            if verbose:
                print(str(datetime.datetime.now()) + ' | applying \
cluster_filter to remove isolated clusters of points in twig_2_mask')
            trunk_2_2_mask = cluster_filter(arr[trunk_2_1], 10, 0.3)
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
            wood_final, leaf_final = continuity_filter(wood, leaf,
                                                       rad=nndist)
        except:
            wood_final = wood
            leaf_final = leaf
    else:
        wood_final = wood
        leaf_final = leaf

    ###########################################################################

    return wood_final, leaf_final


def large_tree_2(arr, class_file=[], knn_lst=[20, 40, 60, 80], cont_filt=True,
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
    knn_lst: list
        Set of knn values to use in the neighborhood search in classification
        steps. This variable will be directly used in a step containing
        the function wlseparate_ref_voting  and its minimum value will be used
        in another step containing wlseparate_abs (both from
        classification.wlseparate). These values are directly dependent of
        point density and were defined based on a medium point density
        scenario (mean distance between points aroun 0.05m). Therefore, for
        higher density point clouds it's recommended the use of larger knn
        values for optimal results.
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

    # Setting up knn value based on the minimum value from knn_lst.
    knn = np.min(knn_lst)

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
        ids_1, prob_1 = wlseparate_abs(arr[not_trunk_ids], knn,
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

        try:
            # Applying class_filter to remove wood_1 points that are more
            # likely to be part of a leaf point cloud (not_wood_1).
            if verbose:
                print(str(datetime.datetime.now()) + ' | \
applying class_filter to remove wood_1 points that are more likely to be \
part of a leaf point cloud (not_wood_1)')
            # Setting up a boolean mask of wood_1 and not_wood_1 points.
            wood_1_bool = np.zeros(arr.shape[0], dtype=bool)
            wood_1_bool[wood_1] = True

            # Obtaining wood_1 filtered point indices.
            if verbose:
                print(str(datetime.datetime.now()) + ' | obtaining wood_1 \
filtered point indices')
            wood_1_1_mask, _ = class_filter(arr[wood_1_bool],
                                            arr[~wood_1_bool], 0, knn=10)
            wood_1_1_mask = np.where(wood_1_1_mask)[0]
            wood_1_1 = wood_1[wood_1_1_mask]
        except:
            wood_1_1 = wood_1

    except:
        # In case absolute threshold separation fails, set wood_1_1 as an
        # empty list.
        wood_1_1 = []
        if verbose:
            print(str(datetime.datetime.now()) + ' | absolute threshold \
separation failed, setting wood_1_1 as an empty list')
    ###########################################################################
    try:
        # Performing reference class voting separation on the whole input point
        # cloud.
        # Running reference class voting separation.
        if verbose:
            print(str(datetime.datetime.now()) + ' | running reference class \
voting separation')
        ids_2, count_2, prob_2 = wlseparate_ref_voting(arr[not_trunk_ids],
                                                       knn_lst, class_file,
                                                       n_classes=4)

        # Obtaining indices and classification probabilities for classes
        # twig and trunk (both components of wood points).
        twig_2_mask = not_trunk_ids[ids_2['twig']]
        twig_2_prob = prob_2['twig']

        # Masking twig and trunk classes by classification probability
        # threshold.
        twig_2_prob_mask = twig_2_prob >= class_prob_threshold

        # Obtaining twig_2 and trunk_2 vote counts, which are the number of
        # votes that each point in twig_2 and trunk_2 received to be
        # classified as such.
        twig_2_count = count_2['twig']

        # Filtering twig_2 and trunk_2 by a minimun number of votes. Point
        # indices with number of votes smaller than the defined threshold
        # are left out.
        twig_2 = twig_2_mask[twig_2_count >= 2][twig_2_prob_mask]

        try:
            # Applying class_filter on filtered twig point cloud.
            if verbose:
                print(str(datetime.datetime.now()) + ' | applying \
class_filter on filtered twig point cloud')
            # Setting up a boolean mask of twig_2 and not_twig_2 points.
            twig_2_bool = np.zeros(arr.shape[0], dtype=bool)
            twig_2_bool[twig_2] = True
            twig_2_1_mask, _ = class_filter(arr[twig_2_bool],
                                            arr[~twig_2_bool], 0, knn=10)
            twig_2_1_mask = np.where(twig_2_1_mask)[0]
            twig_2_1 = twig_2[twig_2_1_mask]

            # Applying radius_filter on filtered twig point cloud.
            if verbose:
                print(str(datetime.datetime.now()) + ' | applying \
radius_filter on filtered twig point cloud')
            twig_2_2_mask = radius_filter(arr[twig_2_1], 0.05, 5)
            twig_2_2 = twig_2_1[twig_2_2_mask]

        except:
            twig_2_2 = twig_2

    except:
        # In case voting separation fails, set twig_2_2 as an empty list.
        twig_2_2 = []
        if verbose:
            print(str(datetime.datetime.now()) + ' | reference class \
separation failed, setting twig_2_2 as an empty list')
    ###########################################################################
    # Stacking all clouds part of the wood portion.
    wood_ids = np.hstack((base_ids, trunk_ids, twig_2_2, wood_1_1))
    wood_ids = np.unique(wood_ids).astype(int)

    # Filtering out tips of each branch, which should remove leaf points
    # misclassified as wood.
    path_filter_mask = detect_main_pathways(arr[wood_ids], 8, 100, .06,
                                            verbose=verbose)
    final_wood = wood_ids[path_filter_mask]

    wood = arr[final_wood]
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
            wood_final, leaf_final = continuity_filter(wood, leaf,
                                                       rad=nndist / 2)
        except:
            wood_final = wood
            leaf_final = leaf
    else:
        wood_final = wood
        leaf_final = leaf

    ###########################################################################

    return wood_final, leaf_final


def large_tree_3(arr, class_file=[], knn_lst=[20, 40, 60, 80], cont_filt=True,
                 class_prob_threshold=0.95, cf_rad=[], verbose=False):

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
    knn_lst: list
        Set of knn values to use in the neighborhood search in classification
        steps. This variable will be directly used in a step containing
        the function wlseparate_ref_voting  and its minimum value will be used
        in another step containing wlseparate_abs (both from
        classification.wlseparate). These values are directly dependent of
        point density and were defined based on a medium point density
        scenario (mean distance between points aroun 0.05m). Therefore, for
        higher density point clouds it's recommended the use of larger knn
        values for optimal results.
    cont_filt : boolean
        Option to select if continuity_filter should be applied to wood and
        leaf point clouds. Default is True.
    class_prob_threshold : float
        Classification probability threshold to filter classes. This aims to
        avoid selecting points that are not confidently enough assigned to
        any given class. Default is 0.95.
    cf_rad : float
        Continuity filter search radius.
    verbose : bool
        Option to set (or not) verbose output.

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

    # Setting up knn value based on the minimum value from knn_lst.
    knn = np.min(knn_lst)

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
        trunk_mask = voxel_path_detection(arr, 0.05, 40, 100, 0.1, True)
        # Obtaining indices of points that are part of the trunk (trunk_ids)
        # and not part of the trunk (not_trunk_ids).
        # trunk.
        trunk_ids = np.where(trunk_mask)[0].astype(int)
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
        ids_1, prob_1 = wlseparate_abs(arr[not_trunk_ids], knn,
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

        try:
            # Applying class_filter to remove wood_1 points that are more
            # likely to be part of a leaf point cloud (not_wood_1).
            if verbose:
                print(str(datetime.datetime.now()) + ' | \
applying class_filter to remove wood_1 points that are more likely to be \
part of a leaf point cloud (not_wood_1)')
            # Setting up a boolean mask of wood_1 and not_wood_1 points.
            wood_1_bool = np.zeros(arr.shape[0], dtype=bool)
            wood_1_bool[wood_1] = True

            # Obtaining wood_1 filtered point indices.
            if verbose:
                print(str(datetime.datetime.now()) + ' | obtaining wood_1 \
filtered point indices')
            wood_1_1_mask, _ = class_filter(arr[wood_1_bool],
                                            arr[~wood_1_bool], 0, knn=10)
            wood_1_1_mask = np.where(wood_1_1_mask)[0]
            wood_1_1 = wood_1[wood_1_1_mask]
        except:
            wood_1_1 = wood_1

    except:
        # In case absolute threshold separation fails, set wood_1_1 as an
        # empty list.
        wood_1_1 = []
        if verbose:
            print(str(datetime.datetime.now()) + ' | absolute threshold \
separation failed, setting wood_1_1 as an empty list')
    ###########################################################################
    try:
        # Performing reference class voting separation on the whole input point
        # cloud.
        # Running reference class voting separation.
        if verbose:
            print(str(datetime.datetime.now()) + ' | running reference class \
voting separation')
        ids_2, count_2, prob_2 = wlseparate_ref_voting(arr[not_trunk_ids],
                                                       knn_lst, class_file,
                                                       n_classes=4)

        # Obtaining indices and classification probabilities for classes
        # twig and trunk (both components of wood points).
        twig_2_mask = not_trunk_ids[ids_2['twig']]
        twig_2_prob = prob_2['twig']

        # Masking twig and trunk classes by classification probability
        # threshold.
        twig_2_prob_mask = twig_2_prob >= class_prob_threshold

        # Obtaining twig_2 and trunk_2 vote counts, which are the number of
        # votes that each point in twig_2 and trunk_2 received to be
        # classified as such.
        twig_2_count = count_2['twig']

        # Filtering twig_2 and trunk_2 by a minimun number of votes. Point
        # indices with number of votes smaller than the defined threshold
        # are left out.
        twig_2 = twig_2_mask[twig_2_count >= 2][twig_2_prob_mask]

        try:
            # Applying class_filter on filtered twig point cloud.
            if verbose:
                print(str(datetime.datetime.now()) + ' | applying \
class_filter on filtered twig point cloud')
            # Setting up a boolean mask of twig_2 and not_twig_2 points.
            twig_2_bool = np.zeros(arr.shape[0], dtype=bool)
            twig_2_bool[twig_2] = True
            twig_2_1_mask, _ = class_filter(arr[twig_2_bool],
                                            arr[~twig_2_bool], 0, knn=10)
            twig_2_1_mask = np.where(twig_2_1_mask)[0]
            twig_2_1 = twig_2[twig_2_1_mask]

            # Applying radius_filter on filtered twig point cloud.
            if verbose:
                print(str(datetime.datetime.now()) + ' | applying \
radius_filter on filtered twig point cloud')
            twig_2_2_mask = radius_filter(arr[twig_2_1], 0.05, 5)
            twig_2_2 = twig_2_1[twig_2_2_mask]

        except:
            twig_2_2 = twig_2

    except:
        # In case voting separation fails, set twig_2_2 as an empty list.
        twig_2_2 = []
        if verbose:
            print(str(datetime.datetime.now()) + ' | reference class \
separation failed, setting twig_2_2 as an empty list')
    ###########################################################################
    # Stacking all clouds part of the wood portion.
    wood_ids = np.hstack((base_ids, trunk_ids, twig_2_2, wood_1_1))
    wood_ids = np.unique(wood_ids).astype(int)

    # Selecting initial set of wood and leaf points.
    wood = arr[wood_ids]

    ###########################################################################

    # Applying path filter to remove small clusters of leaves at the tips of
    # the branches.
    if verbose:
         print(str(datetime.datetime.now()) + ' | running path filtering \
on wood points')
    try:
        path_filter_mask = voxel_path_detection(wood, 0.05, 8, 100, 0.1,
                                                verbose=True)
        wood_filt_1 = wood[path_filter_mask]
        leaf_filt_1 = get_diff(arr, wood_filt_1)
    except:
        if verbose:
             print(str(datetime.datetime.now()) + ' | failed running path \
filtering')
        wood_filt_1 = wood
        leaf_filt_1 = get_diff(arr, wood_filt_1)
    ###########################################################################
    if cont_filt:
        # Applying continuity filter in an attempt to close gaps in the wood
        # point cloud (i.e. misclassified leaf points in between portions of
        # wood points).
        if verbose:
            print(str(datetime.datetime.now()) + ' | applying continuity \
filter in an attempt to close gaps in the wood point cloud')
        try:
            wood_filt_2, leaf_filt_2 = continuity_filter(wood_filt_1,
                                                         leaf_filt_1,
                                                         rad=nndist)

            # Applying path filter agin to clean up data after continuity filter.
            if verbose:
                 print(str(datetime.datetime.now()) + ' | running path \
filtering on wood points')
            try:
                path_filter_mask_2 = voxel_path_detection(wood_filt_2, 0.05,
                                                          4, 100, 0.1,
                                                          verbose=True)
                wood_filt_2 = wood_filt_2[path_filter_mask_2]
            except:
                if verbose:
                     print(str(datetime.datetime.now()) + ' | failed running \
path filtering')
                wood_filt_2 = wood_filt_2

        except:
            wood_filt_2 = wood_filt_1
    else:
        wood_filt_2 = wood_filt_1

    ###########################################################################
    # After filtering wood points, add back smaller branches to fill in
    # the tips lost by path filtering.
    wood_final = np.vstack((wood_filt_2, arr[wood_1_1]))
    wood_final = remove_duplicates(wood_final)
    # Obtaining leaf point cloud from the difference between input cloud 'arr'
    # and wood points.
    leaf_final = get_diff(arr, wood_final)

    ###########################################################################

    return wood_final, leaf_final
