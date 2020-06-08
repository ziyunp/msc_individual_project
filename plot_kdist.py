from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt

def plot_kdist(distance, k = 4):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='precomputed').fit(distance)
    distances, indices = nbrs.kneighbors(distance)
    kth_dist = distances[:,-1]
    kth_dist.sort()
    x = indices[:,0]
    y = kth_dist
    plt.plot(x, y)
    plt.show()

