"""
% Eli Billauer, 3.4.05 (Explicitly not copyrighted).
% This function is released to the public domain; Any use is allowed.

Modifications in docstrings were performed by TLSepartion project
to improve autodocumentation using Sphinx. All credits are still to
Eli Billauer.

"""

import sys
import numpy as np


def peakdet(v, delta, x=None):

    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html


    Parameters
    ----------
    v: array
        Input vector (1D array) of values.
    delta: float
        Value change that characterizes a peak. A point is considered a
        maximum peak if it has the maximal value, and was preceded
        (to the left) by a value lower by delta.
    x: array
        Set of x values to replace indices in maxtab/mintab.

    Returns
    -------
    maxtab: array
        2D array containing maxima peaks indices and values.
    mintab: array
        2D array containing minima peaks indices and values.

    Notes
    ----------
    Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    This function is released to the public domain; Any use is allowed.

    """
    maxtab = []
    mintab = []

    if x is None:
        x = np.arange(len(v))

    v = np.asarray(v)

    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')

    if not np.isscalar(delta):
        sys.exit('Input argument delta must be a scalar')

    if delta <= 0:
        sys.exit('Input argument delta must be positive')

    mn, mx = np.Inf, -np.Inf
    mnpos, mxpos = np.NaN, np.NaN

    lookformax = True

    for i in np.arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]

        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return np.array(maxtab), np.array(mintab)
