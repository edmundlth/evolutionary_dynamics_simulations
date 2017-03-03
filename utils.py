import scipy 
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
import time

def random_exp_time(n, rate=1):
    """
    Return a random vector with n entries, 
    each beingn a waiting time which is exponentially
    distributed.
    """
    if rate <= 0:
        return np.array([np.inf])
    return -1.0/rate * np.log(np.random.rand(n))

def nearest_neighbor_walk(n, p, h):
    """
    Return a n-array of {0, -1, 1} with probability p being not 0, 
    and a bias of h (h=1 has no bias while h=0 means always -1).
    """
    rand_vec = np.random.rand(n)
    p_left = p / (1.0 + h)
    p_right = p * h / (1.0 + h)
    move_prob = np.greater(p_left + p_right, rand_vec)
    move_left_prob = np.greater(p_left, rand_vec)
    return move_prob - 2 * move_prob * move_left_prob

def compute_bins(series):
    """
    Friedman-Diaconis rule of thumb for histogram bin number
    """
    iqr = stats.iqr(series)
    n = series.size
    bins = (series.max() - series.min()) / (2 * iqr * n ** (-1/3.0))
    #print "iqr, n, bins, max, min"
    #print iqr, n, bins, series.max(), series.min(), (np.isnan(bins) or np.isinf(bins))
    if np.isnan(bins) or np.isinf(bins):
        return 10 # return default
    return int(bins)
