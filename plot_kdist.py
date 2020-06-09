from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
import logging
import os

log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO")) #added

def plot_kdist(distance, figName, k = 3):
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

