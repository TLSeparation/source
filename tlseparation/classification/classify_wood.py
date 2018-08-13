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

import numpy as np
from classes_reference import DefaultClass
from wlseparation import wlseparate_abs, wlseparate_ref_voting
from ..utility.filtering import class_filter


def reference_classification(point_cloud, knn_list, n_classes=4,
                             prob_threshold=0.95):

    """
    Classifies wood material points from a point cloud. This function
    uses *wlseparate_ref_voting* to perform the basic classification and then
    apply *class_filter* to filter out potentially misclassified wood points.

    Parameters
    ----------
    point_cloud: numpy.ndarray
        2D (n x 3) array containing n points in 3D space (x, y, z).
    knn_list: list
        List of knn values to be used iteratively in the voting separation.
    n_classes: int
        Number of intermediate classes. Minimum classes should be 3, but
        default value is set to 4 in order to accommodate for noise/outliers
        classes.
    prob_threshold: float
        Classification probability threshold to filter classes. This aims to
        avoid selecting points that are not confidently enough assigned to
        any given class. Default is 0.95.

    Returns
    -------
    wood_points: numpy.ndarray
        2D (nw x 3) array containing n wood points in 3D space (x, y, z).

    """

    # Defining reference class table.
    class_file = DefaultClass().ref_table

    # Classifying point cloud using wlseparate_ref_voting. The output will
    # be a combination of classes indices, vote counts and probabilities.
    ids, count, prob = wlseparate_ref_voting(point_cloud, knn_list, class_file,
                                             n_classes=n_classes)
    # Selecting indices, probabilities and votes count for wood classes
    # (twig and trunk).
    twig_mask = ids['twig']
    twig_prob = prob['twig']
    twig_count = count['twig']
    # Selecting only twig points with a high probability and vote count.
    twig = twig_mask[(twig_prob >= prob_threshold) &
                     (twig_count >= np.max(twig_count) - 1)]
    trunk_mask = ids['trunk']
    trunk_prob = prob['trunk']
    trunk_count = count['trunk']
    # Selecting only trunk points with a high probability and vote count.
    trunk = trunk_mask[(trunk_prob >= prob_threshold) &
                       (trunk_count >= np.max(trunk_count) - 1)]

    # Creating boolean mask with the same number of entries as input
    # point cloud. Entries of points classified as wood are set to True.
    class_mask = np.zeros(point_cloud.shape[0], dtype=bool)
    class_mask[twig] = True
    class_mask[trunk] = True

    # Stacking wood and not wood points and applying class_filter.
    temp_arr = np.vstack((point_cloud[class_mask], point_cloud[~class_mask]))
    k = int(np.min(knn_list))
    wood_ids, not_wood_ids = class_filter(point_cloud[class_mask],
                                          point_cloud[~class_mask], 0, knn=k)

    return temp_arr[wood_ids]


def threshold_classification(point_cloud, knn, n_classes=3,
                             prob_threshold=0.95):

    """
    Classifies wood material points from a point cloud. This function
    uses *wlseparate_abs* to perform the basic classification and then
    apply *class_filter* to filter out potentially misclassified wood points.

    Parameters
    ----------
    point_cloud : numpy.ndarray
        2D (n x 3) array containing n points in 3D space (x, y, z).
    knn : int
        Number of neighbors to select around each point. Used to describe
        local point arrangement.
    n_classes: int
        Number of intermediate classes. Default is 3.
    prob_threshold: float
        Classification probability threshold to filter classes. This aims to
        avoid selecting points that are not confidently enough assigned to
        any given class. Default is 0.95.

    Returns
    -------
    wood_points: numpy.ndarray
        2D (nw x 3) array containing n wood points in 3D space (x, y, z).

    """

    # Running wlseparate_abs to classify the input point cloud into wood and
    # leaf classes.
    ids, prob = wlseparate_abs(point_cloud, knn, n_classes)
    # Selecting wood indices and probabilities.
    wood_mask = ids['wood']
    wood_prob = prob['wood']
    # Filtering out wood points with classification probability lower than
    # threshold.
    wood = wood_mask[wood_prob >= prob_threshold]

    # Creating boolean mask with the same number of entries as input
    # point cloud. Entries of points classified as wood are set to True.
    class_mask = np.zeros(point_cloud.shape[0], dtype=bool)
    class_mask[wood] = True

    # Stacking wood and not wood points and applying class_filter.
    temp_arr = np.vstack((point_cloud[class_mask],
                          point_cloud[~class_mask]))
    wood_ids, not_wood_ids = class_filter(point_cloud[class_mask],
                                          point_cloud[~class_mask], 0,
                                          knn=int(knn))

    return temp_arr[wood_ids]
