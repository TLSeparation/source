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
from knnsearch import set_nbrs_knn


def get_diff(arr1, arr2):

    """
    Performs the intersection of two arrays, returning the entries not
    intersected between arr1 and arr2.

    Parameters
    ----------
    arr1 : array
        N-dimensional array of points to intersect.
    arr2 : array
        N-dimensional array of points to intersect.

    Returns
    -------
    arr : array
        Difference array between 'arr1' and 'arr2'.

    """

    # Asserting that both arrays have the same number of columns.
    assert arr1.shape[1] == arr2.shape[1]

    # Stacking both arrays.
    arr3 = np.vstack((arr1, arr2))

    # Creating a pandas.DataFrame from the stacked array.
    df = pd.DataFrame(arr3)

    # Removing duplicate points and keeping only points that have only a
    # single occurrence in the stacked array.
    diff = df.drop_duplicates(keep=False)

    return np.asarray(diff)


def remove_duplicates(arr, return_ids=False):

    """
    Removes duplicated rows from an array.

    Parameters
    ----------
    arr : array
        N-dimensional array (m x n) containing a set of parameters (n) over
        a set of observations (m).
    return_ids: bool
        Option to return indices of duplicated entries instead of new array
        with unique entries.

    Returns
    -------
    unique : array
        N-dimensional array (m* x n) containing a set of unique parameters (n)
        over a set of unique observations (m*).

    """

    # Setting the pandas.DataFrame from the array (arr) data.
    df = pd.DataFrame({'x': arr[:, 0],
                       'y': arr[:, 1], 'z': arr[:, 2]})

    if return_ids:
        # Using the duplicated function to mask duplicate points from df.
        return np.where(df.duplicated((['x', 'y', 'z'])))[0]

    else:
        # Using the drop_duplicates function to remove duplicate points
        # from df.
        unique = df.drop_duplicates(['x', 'y', 'z'])

        return np.asarray(unique).astype(float)


def apply_nn_value(base, arr, attr):

    """
    Upscales a set of attributes from a base array to another denser array.

    Parameters
    ----------
    base : array
        Base array to which the attributes to upscale were originaly matched.
    arr : array
        Target array to which the attributes will be upscaled.
    attr : array
        Attributes to upscale.

    Returns
    -------
    new_attr : array
        Upscales attributes.

    Raises
    ------
    AssertionError:
        length (number of samples) of "base" and "attr" must be equal.

    """

    assert base.shape[0] == attr.shape[0], '"base" and "attr" must have the\
 same number of samples.'

    # Obtaining the closest in base for each point in arr.
    idx = set_nbrs_knn(base, arr, 1, return_dist=False)

    # Making sure idx has the right type, int, for indexing.
    idx = idx.astype(int)

    # Applying base's attribute (attr) to points in arr.
    newattr = attr[idx]

    return np.reshape(newattr, newattr.shape[0])


def entries_to_remove(entries, d):

    """
    Function to remove selected entries (key and respective values) from
    a given dict.
    Based on a reply from the user mattbornski [#]_ at stackoverflow.

    Parameters
    ----------
    entries : array
        Set of entried to be removed.
    d : dict
        Dictionary to apply the entried removal.

    References
    ----------
    ..  [#] mattbornski, 2012. http://stackoverflow.com/questions/8995611/\
removing-multiple-keys-from-a-dictionary-safely

    """

    for k in entries:
        d.pop(k, None)
