import numpy as np


def z_score(x):
    """
    Normilises a set of values by calculating the Z-score.

    Parameters
    ----------
    x : pandas.Series, numpy.Array, list of other similar iterable object
        The value series to be normalised.
   
    Returns
    -------
    Iterable object inputted
        The value series normilised using Z-scoring.
    """
    return (x - np.mean(x)) / np.std(x)
