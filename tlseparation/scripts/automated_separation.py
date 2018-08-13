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

import numpy as np
import datetime
from ..classification import (wlseparate_abs, wlseparate_ref_voting,
                              threshold_classification,
                              reference_classification, path_detect_frequency,
                              voxel_path_detection, get_base, DefaultClass)
from ..utility import (get_diff, remove_duplicates, radius_filter,
                       class_filter, cluster_filter, continuity_filter,
                       feature_filter, plane_filter,
                       detect_nn_dist)


def large_tree_3(arr, class_file=[], knn_lst=[20, 40, 60, 80], gmm_nclasses=4,
                 class_prob_threshold=0.95, cont_filt=True, cf_rad=None,
                 verbose=False):

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
    gmm_nclasses: int
        Number of classes to use in Gaussian Mixture Classification. Default
        is 4.
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
    # Checking if no input was given to cf_rad and if so, calculate it from
    # nndist.
    if cf_rad is None:
        cf_rad = nndist * 0.66

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
        trunk_mask = voxel_path_detection(arr, 0.1, 40, 100, 0.15, True)
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
                                       n_classes=gmm_nclasses)

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
                                                       n_classes=gmm_nclasses)

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
        path_filter_mask = voxel_path_detection(wood, 0.1, 8, 100, 0.15,
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
                                                         rad=cf_rad)

            # Applying path filter agin to clean up data after continuity filter.
            if verbose:
                 print(str(datetime.datetime.now()) + ' | running path \
filtering on wood points')
            try:
                path_filter_mask_2 = voxel_path_detection(wood_filt_2, 0.1,
                                                          4, 100, 0.15,
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


def large_tree_4(arr, class_file=[], knn_lst=[20, 40, 60, 80], gmm_nclasses=4,
                 class_prob_threshold=0.95, cont_filt=True, cf_rad=None,
                 verbose=False):

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
    gmm_nclasses: int
        Number of classes to use in Gaussian Mixture Classification. Default
        is 4.
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
    # Checking if no input was given to cf_rad and if so, calculate it from
    # nndist.
    if cf_rad is None:
        cf_rad = nndist * 0.66

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
        trunk_mask = voxel_path_detection(arr, 0.1, 40, 100, 0.15, True)
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
                                       n_classes=gmm_nclasses)

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
                                                       n_classes=gmm_nclasses)

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

    mask_plane = plane_filter(wood, 0.05, 0.02)
    mask_feature = feature_filter(wood, 4, -1, 30)
    temp_mask = np.logical_and(mask_plane, mask_feature)
    mask_cluster = cluster_filter(wood, 0.05, 0.2)
    final_mask = np.logical_and(temp_mask, mask_cluster)
    wood = wood[final_mask]
    leaf = get_diff(arr, wood)


    ###########################################################################

    # Applying path filter to remove small clusters of leaves at the tips of
    # the branches.
    if verbose:
         print(str(datetime.datetime.now()) + ' | running path filtering \
on wood points')
    try:
        path_filter_mask = voxel_path_detection(wood, 0.1, 8, 100, 0.15,
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
                                                         rad=cf_rad)

            # Applying path filter agin to clean up data after continuity filter.
            if verbose:
                 print(str(datetime.datetime.now()) + ' | running path \
filtering on wood points')
            try:
                path_filter_mask_2 = voxel_path_detection(wood_filt_2, 0.1,
                                                          4, 100, 0.15,
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


def generic_tree(arr, knn_list=[40, 50, 80, 120, 100], voxel_size=0.05,
                 retrace_steps=40):

    """
    Run an automated separation of a single tree point cloud.

    Parameters
    ----------
    arr : array
        Three-dimensional point cloud of a single tree to perform the
        wood-leaf separation. This should be a n-dimensional array (m x n)
        containing a set of coordinates (n) over a set of points (m).
    knn_lst: list
        Set of knn values to use in the neighborhood search in classification
        steps. This variable will be directly used in a step containing
        the function reference_classification  and its minimum and maximum
        values will be used in a different step with threshold_classification
        (both from classification.classify_wood). These values are directl
        dependent of point density and were defined based on a medium point
        density scenario (mean distance between points aroun 0.05m).
        Therefore, for higher density point clouds it's recommended the use of
        larger knn values for optimal results.
    verbose : bool
        Option to set (or not) verbose output.

    Returns
    -------
    wood_final : array
        Wood point cloud.
    leaf_final : array
        Leaf point cloud.

    """

    # Running voxel_path_detection to detect main pathways (trunk and
    # low order branches) in a tree point cloud. This step generates a
    # graph from the point cloud and retrace n retrace_steps towards the
    # root of the tree.
    path_mask = voxel_path_detection(arr, voxel_size, retrace_steps, 100,
                                     voxel_size * 1.77, False)
    # Filtering path_mask points by feature threshold. In this case,
    # feature 4 has a very distinctive pattern for wood and leaf. Usually
    # the threshold is around -0.9 to -1.
    path_mask_feature = feature_filter(arr[path_mask], 4, -0.9,
                                       np.min(knn_list))
    # Selecting filtered points in path_mask.
    path_retrace_arr = arr[path_mask][path_mask_feature]
    # Running path_detect_frequency to detect main pathways (trunk and
    # low order branches) in a tree point cloud. This step generates a
    # graph from the point cloud and select nodes with high frequency
    # of paths passing through.
    path_frequency_arr = path_detect_frequency(arr, 0.05, 6)
    # Running threshold_classification to detect small branches.
    wood_abs = threshold_classification(arr, np.min(knn_list))
    # Running reference_classification to detect both trunk, medium branches
    # and small branches.
    wood_vote = reference_classification(arr, knn_list)
    # Stacking classified wood points.
    wood1 = np.vstack((wood_abs, wood_vote))
    # Obtaining leaf points by the difference set between wood and initial
    # point clouds.
    leaf1 = get_diff(arr, wood1)
    # Obtaining larger branches that might have been missed in previous
    # steps. The basic idea is to use a much larger knn value.
    wood_abs_2 = threshold_classification(leaf1, np.max(knn_list) * 2)
    # If wood_abs_2 has more than 10 points, do a cluster filtering to
    # remove cluster with round/flat shapes.
    if len(wood_abs_2) >= 10:
        mask_cluster_2 = cluster_filter(wood_abs_2, 0.06, 0.6)
        wood_abs_2 = wood_abs_2[mask_cluster_2]
    # Obtaining small branches that might have been missed in previous
    # steps. To detect small features, the ideal approach is to use a
    # small neighborhood of points.
    wood_abs_3 = threshold_classification(leaf1, np.min(knn_list))
    # Stacking all wood points classified through Gaussian Mixture/EM.
    wood2 = np.vstack((wood1, wood_abs_2, wood_abs_3))
    # Removing duplicated points.
    wood2 = remove_duplicates(wood2)
    # Applying plane filter to remove points in a plane-ish neighborhood
    # of points. These plane points are more likely to be part of leaves.
    mask_plane = plane_filter(wood2, 0.03, 0.02)
    # Stacking final wood points from GMM classification and path
    # detection.
    wood_final = np.vstack((path_frequency_arr, path_retrace_arr,
                            wood2[mask_plane]))
    # Removes duplicate points and obtains final leaf points from
    # the difference set between initial and final wood point clouds.
    wood_final = remove_duplicates(wood_final)
    leaf_final = get_diff(arr, wood_final)

    return wood_final, leaf_final
