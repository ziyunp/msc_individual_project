from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
from kneed import KneeLocator

log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO")) #added

def locate_elbow(distance, figName, k=3):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='precomputed').fit(distance)
    distances, indices = nbrs.kneighbors(distance)
    kth_dist = distances[:,-1]
    kth_dist.sort()
    x_data = indices[:,0]
    y_data = kth_dist
    # Find elbows recursively until x contains less than 4 points (min points to form an elbow)
    x_start = 0
    elbows = []
    elbows_x = []
    while x_start != None and x_start < len(x_data) - 4:
        x = x_data[x_start:]
        y = y_data[x_start:]
        kneedle = KneeLocator(x, y, curve='convex', direction='increasing', interp_method='polynomial')
        # TODO: should we take the last value in y as an elbow value? - the furthest distance from kth neighbours => will include all points
        if kneedle.elbow_y != None:
            elbows.append(kneedle.elbow_y)
        x_start = kneedle.elbow
        elbows_x.append(kneedle.elbow)
        # kneedle.plot_knee()
        # plt.show()

    # Plot all knees
    kl = KneeLocator(x_data, y_data, curve='convex', direction='increasing', interp_method='polynomial')
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(6, 6))
    plt.title("Default")
    plt.plot(kl.x, kl.y, "b", label="data")
    colors = ['r', 'g', 'k', 'm', 'c', 'orange']
    for k, c, s in zip(elbows_x, colors, elbows):
        plt.vlines(k, plt.ylim()[0], plt.ylim()[1], linestyles='--', colors=c, label=f'eps = {s}')
    plt.legend(loc="best")
    fig.savefig(figName, dpi=fig.dpi)
    return elbows