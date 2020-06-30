from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
from kneed import KneeLocator

log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO")) #added

def save_kdist_plot(distance, figName, k = 3):
    log.info("Plotting {}...".format(figName))
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='precomputed').fit(distance)
    distances, indices = nbrs.kneighbors(distance)
    kth_dist = distances[:,-1]
    kth_dist.sort()
    x = indices[:,0]
    y = kth_dist
    fig = plt.figure()
    plt.plot(x, y)
    fig.savefig(figName, dpi=fig.dpi)
    log.info("Saved k-dist plot to {}".format(figName))  

def locate_elbow(distance, figName, k=3):
    log.info("Locating elbow...")
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='precomputed').fit(distance)
    distances, indices = nbrs.kneighbors(distance)
    kth_dist = distances[:,-1]
    kth_dist.sort()
    x = indices[:,0]
    y = kth_dist
    kneedle = KneeLocator(x, y, curve='convex', direction='increasing', interp_method='polynomial')
    kneedle.plot_knee()
    return kneedle.elbow_y