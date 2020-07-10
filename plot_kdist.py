from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
from kneed import KneeLocator

log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO")) #added

def plot_kdist(x, y):
    return KneeLocator(x, y, curve='convex', direction='increasing', interp_method='polynomial')

def locate_elbow(distance, figName, k=4, multiple=False):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='precomputed').fit(distance)
    distances, indices = nbrs.kneighbors(distance)
    kth_dist = distances[:,-1]
    kth_dist.sort()
    x_data = indices[:,0]
    y_data = kth_dist

    elbows = []

    if not multiple:
        kneedle = plot_kdist(x_data, y_data)
        if kneedle.elbow_y != None:
            elbows.append(kneedle.elbow_y)
    else:     
        # Find multiple elbows 
        x_start = 0
        # Min num of points to form an elbow = 3
        while x_start != None and x_start < len(x_data) - 3:
            x = x_data[x_start:]
            y = y_data[x_start:]
            kneedle = plot_kdist(x, y)
            if kneedle.elbow_y != None and kneedle.elbow_y != y_data[-1]:
                if elbows and kneedle.elbow_y / elbows[-1] < 2:
                    # If this elbow is less than double the previous elbow, take the later elbow
                    elbows[-1] = kneedle.elbow_y
                else:
                    elbows.append(kneedle.elbow_y)
            x_start = kneedle.elbow

    # Plot all knees
    kl = plot_kdist(x_data, y_data)
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(6, 6))
    plt.title("Default")
    plt.plot(kl.x, kl.y, "b", label="data")
    colors = ['r', 'g', 'k', 'm', 'c', 'orange']
    for k, c, s in zip(elbows, colors, elbows):
        plt.hlines(k, plt.xlim()[0], plt.xlim()[1], linestyles='--', colors=c, label=f'eps = {s}')
    plt.legend(loc="best")
    fig.savefig(figName, dpi=fig.dpi)
    return elbows