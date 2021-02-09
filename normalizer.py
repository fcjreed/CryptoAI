import math
import numpy as np
import scipy
import statistics

def normalize_zscore(input):
    mean = statistics.mean(input)
    std = statistics.pstdev(input)
    store = []
    for dat in input:
        store.append(dat - mean / std)
    return store

def normalize_min_max(input):
    min_val = min(input)
    max_val = max(input)
    store = []
    for dat in input:
        store.append(dat - min_val / max_val - min_val)
    return store

def normalize_logistic(input):
    store = []
    for dat in input:
        store.append(1 / 1 + math.exp(-dat))
    return store

def normalize_lognormal(input):
    mean = statistics.mean(input)
    shape = statistics.pstdev(np.log10(input))
    scale = np.exp(mean)
    loc = 0
    store = []
    for dat in input:
        store.append(scipy.stats.lognorm.cdf(dat,shape,loc,scale))
    return store

def normalize_tanh(input):
    return np.tanh(input)