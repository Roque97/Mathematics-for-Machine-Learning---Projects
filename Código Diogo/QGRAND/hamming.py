"""Functions for the code's Hamming bound.

Defines functions related to computing the Hamming bound for the code.
"""

import numpy as np
from scipy.special import comb

def shannon_entropy_bernoulli(p):
    """
    Shannon entropy, in base 2, for a Bernoulli distribution.

    Parameters
    ----------
    p : float
        Probability.

    Returns
    -------
    float
        Shannon entropy.
    """

    entropy = -np.nan_to_num(p*np.log2(p)) - np.nan_to_num((1-p)*np.log2(1-p))
    return entropy

def hamming_bound(x):
    return 1 - x*np.log2(3) - shannon_entropy_bernoulli(x)

def asympt_max_distance_hamming(n,k, hamming_f):
    return int(2*n*hamming_f(k/n) + 1)

def asympt_max_distance_varshamov(n,k, hamming_f):
    return int(n*hamming_f(k/n))

def max_distance_hamming(n, k, haming_f=None):
    try:
        s = 0.
        for t in range(int((n-k)/2) + 1):
            s += comb(n, t) * 3**t / 2**(n-k)
            if s > 1:
                return 2*(t-1) + 1
    except OverflowError:
        asympt_max_distance_hamming(n, k, haming_f)