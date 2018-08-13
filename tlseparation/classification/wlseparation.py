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
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from ..utility.knnsearch import set_nbrs_knn
from ..classification.point_features import knn_features
from ..classification.gmm import (classify, class_select_abs,
                                  class_select_ref)


def fill_class(arr1, arr2, noclass, k):

    """
    Assigns noclass entries to either arr1 or arr2, depending on
    neighborhood majority analisys.

    Parameters
    ----------
    arr1 : array
        Point coordinates for entries of the first class.
    arr2 : array
        Point coordinates for entries of the second class.
    noclass : array
        Point coordinates for noclass entries.
    k : int
        Number of neighbors to use in the neighborhood majority analysis.

    Returns
    -------
    arr1 : array
        Point coordinates for entries of the first class.
    arr2 : array
        Point coordinates for entries of the second class.

    """

    # Stacking arr1 and arr2. This will be fitted in the NearestNeighbors
    # search in order to define local majority and assign classes to
    # noclass.
    arr = np.vstack((arr1, arr2))

    # Generating classes labels with the same shapes as arr1, arr2 and,
    # after stacking, arr.
    class_1 = np.full(arr1.shape[0], 1, dtype=np.int)
    class_2 = np.full(arr2.shape[0], 2, dtype=np.int)
    classes = np.hstack((class_1, class_2)).T

    # Performin NearestNeighbors search to detect local sets of points.
    nbrs = NearestNeighbors(leaf_size=25, n_jobs=-1).fit(arr)
    indices = nbrs.kneighbors(noclass, n_neighbors=k, return_distance=False)

    # Allocating output variable.
    new_class = np.zeros(noclass.shape[0])

    # Selecting subset of classes based on the neighborhood expressed by
    # indices.
    class_ = classes[indices]

    # Looping over all points in indices.
    for i in range(len(indices)):

        # Counting the number of occurrences of each value in the ith instance
        # of class_.
        unique, count = np.unique(class_[i, :], return_counts=True)
        # Appending the majority class into the output variable.
        new_class[i] = unique[np.argmax(count)]

    # Stacking new points to arr1 and arr2.
    arr1 = np.vstack((arr1, noclass[new_class == 1]))
    arr2 = np.vstack((arr2, noclass[new_class == 2]))

    # Making sure all points were processed and assigned a class.
    assert ((arr1.shape[0] + arr2.shape[0]) ==
            (arr.shape[0] + noclass.shape[0]))

    return arr1, arr2


def wlseparate_ref_voting(arr, knn_lst, class_file, n_classes=3):

    """
    Classifies a point cloud (arr) into two main classes, wood and leaf.
    Altough this function does not output a noclass category, it still
    filters out results based on classification confidence interval in the
    voting process (if lower than prob_threshold, then voting is not used
    for current point and knn value).

    The final class selection is based a voting scheme applied to a similar
    approach of wlseparate_ref. In this case, the function iterates over a
    series of knn values and apply the reference distance criteria to select
    wood and leaf classes.

    Each knn class result is accumulated in a list and in the end a voting
    is applied. For each point, if the number of times it was classified as
    wood is larger than threhsold, the final class is set to wood. Otherwise
    it is set as leaf.

    Class selection will mask points according to their class mean distance
    to reference classes. The closes reference class gets assignes to each
    intermediate class.

    Parameters
    ----------
    arr : array
        Three-dimensional point cloud of a single tree to perform the
        wood-leaf separation. This should be a n-dimensional array (m x n)
        containing a set of coordinates (n) over a set of points (m).
    knn_lst : list
        List of knn values to use in the search to constitue local subsets of
        points around each point in 'arr'. It can be a single knn value, as
        long as it has list data type.
    class_file : pandas dataframe or str
        Dataframe or path to reference classes file.
    n_classes : int
        Number of classes to use in the Gaussian Mixture Classification.

    Returns
    -------
    class_dict : dict
        Dictionary containing indices for all classes in class_ref. Classes
        are labeled according to classes names in class_file.
    count_dict : dict
        Dictionary containin votes count for all classes in class_ref. Classes
        are labeled according to classes names in class_file.
    prob_dict : dict
        Dictionary containing probabilities for all classes in class_ref.
        Classes are labeled according to classes names in class_file.

    """

    # Making sure 'knn_lst' is of list type.
    if type(knn_lst) != list:
        knn_lst = [knn_lst]

    # Initializing voting accumulator and class probability arrays.
    vt = np.full([arr.shape[0], len(knn_lst)], -1, dtype=int)
    prob = np.full([arr.shape[0], len(knn_lst)], -1, dtype=float)

    # Generating a base set of indices and distances around each point.
    # This step uses the largest value in knn_lst to make further searches,
    # with smaller values of knn, more efficient.
    idx_base = set_nbrs_knn(arr, arr, np.max(knn_lst), return_dist=False)

    # Reading in class reference values from file.
    if isinstance(class_file, str):
        class_table = pd.read_csv(class_file)
        print class_table
    elif isinstance(class_file, pd.core.frame.DataFrame):
        class_table = class_file
    else:
        raise Exception('class file should be a pandas dataframe or file path')
    class_ref = np.asarray(class_table.iloc[:, 1:]).astype(float)

    # Looping over values of knn in knn_lst.
    for i, k in enumerate(knn_lst):
        # Subseting indices and distances based on initial knn search and
        # current knn value (k).
        idx_1 = idx_base[:, :k+1]

        # Calculating the geometric descriptors.
        gd_1 = knn_features(arr, idx_1)

        # Classifying the points based on the geometric descriptors.
        classes_1, cm_1, proba_1 = classify(gd_1, n_classes)
        cm_1 = ((cm_1 - np.min(cm_1, axis=0)) /
                (np.max(cm_1, axis=0) - np.min(cm_1, axis=0)))

        # Selecting which classes represent classes from classes reference
        # file.
        new_classes = class_select_ref(classes_1, cm_1, class_ref)

        # Appending results to vt temporary list.
        vt[:, i] = new_classes.astype(int)
        prob[:, i] = np.max(proba_1, axis=1)

    # Performing the voting scheme (majority selection) for each point.
    # Initializing final_* variables to store class number, vote counts and
    # class provability.
    final_class = np.full([arr.shape[0]], -1, dtype=int)
    final_count = np.full([arr.shape[0]], -1, dtype=int)
    final_prob = np.full([arr.shape[0]], -1, dtype=float)
    # Iterating over class votes (vt) and their probabilities (prob).
    for i, (v, p) in enumerate(zip(vt, prob)):
        # Counting votes of each class.
        unique, count = np.unique(v, return_counts=True)
        # Appending to final_* arrays the most voted class, the total number
        # of votes this class received and it's classficiation probability.
        final_class[i] = unique[np.argmax(count)]
        final_count[i] = count[np.argmax(count)]
        # Masking entries that received a vote for the most voted class.
        final_class_mask = v == final_class[i]
        # Averaging over all classification probabilities for all votes of
        # the most voted class.
        final_prob[i] = np.mean(p[final_class_mask])

    # Selecting classes labels from entries in class_ref.
    # Generating indices array to help in future indexing.
    idx = np.arange(arr.shape[0], dtype=int)
    # Initializing dictionaires for output variables.
    class_dict = {}
    count_dict = {}
    prob_dict = {}
    # Looping over each unique class in final_class.
    for c in np.unique(final_class).astype(int):
        # Selecting all indices for points that were classfied as
        # belonging to current class.
        class_idx = idx[final_class == c]
        # Selecting all vote counts for points that were classfied as
        # belonging to current class. Only gets votes of most voted class for
        # each point.
        class_count = final_count[final_class == c]
        # Selecting all classification probabilities for points that were
        # classfied as belonging to current class. Only gets probability of
        # most voted class for each point.
        class_prob = final_prob[final_class == c]
        # Assigining current class indices, votes and probability to
        # output dictionaries. Current key name is set as selected class name
        # from class_ref.
        class_dict[class_table.iloc[c, :]['class']] = class_idx
        count_dict[class_table.iloc[c, :]['class']] = class_count
        prob_dict[class_table.iloc[c, :]['class']] = class_prob

    return class_dict, count_dict, prob_dict


def wlseparate_abs(arr, knn, knn_downsample=1, n_classes=3):

    """
    Classifies a point cloud (arr) into three main classes, wood, leaf and
    noclass.

    The final class selection is based on the absolute value of the last
    geometric feature (see point_features module).
    Points will be only classified as wood or leaf if their classification
    probability is higher than prob_threshold. Otherwise, points are
    assigned to noclass.

    Class selection will mask points with feature value larger than a given
    threshold as wood and the remaining points as leaf.

    Parameters
    ----------
    arr : array
        Three-dimensional point cloud of a single tree to perform the
        wood-leaf separation. This should be a n-dimensional array (m x n)
        containing a set of coordinates (n) over a set of points (m).
    knn : int
        Number of nearest neighbors to search to constitue the local subset of
        points around each point in 'arr'.
    knn_downsample : float
        Downsample factor (0, 1) for the knn parameter. If less than 1, a
        sample of size (knn * knn_downsample) will be selected from the
        nearest neighbors indices. This option aims to maintain the spatial
        representation of the local subsets of points, but reducing overhead
        in memory and processing time.
    n_classes : int
        Number of classes to use in the Gaussian Mixture Classification.

    Returns
    -------
    class_indices : dict
        Dictionary containing indices for wood and leaf classes.
    class_probability : dict
        Dictionary containing probabilities for wood and leaf classes.

    """

    # Generating the indices array of the 'k' nearest neighbors (knn) for all
    # points in arr.
    idx_1 = set_nbrs_knn(arr, arr, knn, return_dist=False)

    # If downsample fraction value is set to lower than 1. Apply downsampling
    # on knn indices.
    if knn_downsample < 1:
        n_samples = np.int(idx_1.shape[1] * knn_downsample)
        idx_f = np.zeros([idx_1.shape[0], n_samples + 1])
        idx_f[:, 0] = idx_1[:, 0]
        for i in range(idx_f.shape[0]):
            idx_f[i, 1:] = np.random.choice(idx_1[i, 1:], n_samples,
                                            replace=False)
        idx_1 = idx_f.astype(int)

    # Calculating geometric descriptors.
    gd_1 = knn_features(arr, idx_1)

    # Classifying the points based on the geometric descriptors.
    classes_1, cm_1, proba_1 = classify(gd_1, n_classes)

    # Selecting which classes represent wood and leaf. Wood classes are masked
    # as True and leaf classes as False.
    mask_1 = class_select_abs(classes_1, cm_1, idx_1)

    # Generating set of indices of entries in arr. This will be part of the
    # output.
    arr_ids = np.arange(0, arr.shape[0], 1, dtype=int)

    # Creating output class indices dictionary and class probabilities
    # dictionary.
    # mask represent wood points, (~) not mask represent leaf points.
    class_indices = {}
    class_probability = {}
    try:
        class_indices['wood'] = arr_ids[mask_1]
        class_probability['wood'] = np.max(proba_1, axis=1)[mask_1]
    except:
        class_indices['wood'] = []
        class_probability['wood'] = []
    try:
        class_indices['leaf'] = arr_ids[~mask_1]
        class_probability['leaf'] = np.max(proba_1, axis=1)[~mask_1]
    except:
        class_indices['leaf'] = []
        class_probability['leaf'] = []

    return class_indices, class_probability
