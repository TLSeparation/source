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
from sklearn.mixture import GaussianMixture as GMM


def classify(variables, n_classes):

    """
    Function to perform the classification of a dataset using sklearn's
    Gaussian Mixture Models with Expectation Maximization.

    Parameters
    ----------
    variables : array
        N-dimensional array (m x n) containing a set of parameters (n)
        over a set of observations (m).
    n_classes : int
        Number of classes to assign the input variables.

    Returns
    -------
    classes : list
        List of classes labels for each observation from the input variables.
    means : array
        N-dimensional array (c x n) of each class (c) parameter space means
        (n).
    probability : array
        Probability of samples belonging to every class in the classification.
        Sum of sample-wise probability should be 1.

    """

    # Initialize a GMM classifier with n_classes and fit variables to it.
    gmm = GMM(n_components=n_classes)
    gmm.fit(variables)

    return gmm.predict(variables), gmm.means_, gmm.predict_proba(variables)


def class_select_ref(classes, cm, classes_ref):

    """
    Selects from the classification results which classes are wood and which
    are leaf.

    Parameters
    ----------
    classes : list
        List of classes labels for each observation from the input variables.
    cm : array
        N-dimensional array (c x n) of each class (c) parameter space mean
        valuess (n).
    classes_ref : array
        Reference classes values.

    Returns
    -------
    mask : array
        List of booleans where True represents wood points and False
        represents leaf points.

    """

    # Initializing array of class ids.
    class_ids = np.zeros([cm.shape[0]])

    # Looping over each index in the classes means array.
    for c in range(cm.shape[0]):
        # Setting initial minimum distance value.
        mindist = np.inf
        # Looping over indices in classes reference values.
        for i in range(classes_ref.shape[0]):
            # Calculating distance of current class mean parameters and
            # current reference paramenters.
            d = np.linalg.norm(cm[c] - classes_ref[i])
            # Checking if current distance is smaller than previous distance
            # if so, assign current reference index to current class index.
            if d < mindist:
                class_ids[c] = i
                mindist = d

    # Assigning final classes values to new classes.
    new_classes = np.zeros([classes.shape[0]])
    for i in range(new_classes.shape[0]):
        new_classes[i] = class_ids[classes[i]]

    return new_classes


def class_select_abs(classes, cm, nbrs_idx, feature=5, threshold=0.5):

    """
    Select from GMM classification results which classes are wood and which
    are leaf based on a absolute value threshold from a single feature in
    the parameter space.

    Parameters
    ----------
    classes : list or array
        Classes labels for each observation from the input variables.
    cm : array
        N-dimensional array (c x n) of each class (c) parameter space mean
        valuess (n).
    nbrs_idx : array
        Nearest Neighbors indices relative to every point of the array that
        originated the classes labels.
    feature : int
        Column index of the feature to use as constraint.
    threshold : float
        Threshold value to mask classes. All classes with means >= threshold
        are masked as true.

    Returns
    -------
    mask : list
        List of booleans where True represents wood points and False
        represents leaf points.

    """

    # Calculating the ratio of first 3 components of the classes means (cm).
    # These components are the basic geometric descriptors.
    if np.max(np.sum(cm, axis=1)) >= threshold:

        class_id = np.argmax(cm[:, feature])

        # Masking classes based on the criterias set above. Mask will present
        # True for wood points and False for leaf points.
        mask = classes == class_id

    else:
        mask = []

    return mask
