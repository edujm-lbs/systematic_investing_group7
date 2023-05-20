import numpy as np


def z_score(x):
    """
    used to calculate z-scores
    """
    return (x - np.mean(x)) / np.std(x)
