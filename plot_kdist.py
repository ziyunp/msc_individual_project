from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
from kneed import KneeLocator

log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO")) #added

def valid_elbow(elbow, last_data_value):
    return elbow != None and elbow > 0 and elbow != last_data_value

def plot_kdist(x, y):
    return KneeLocator(x, y, curve='convex', direction='increasing', interp_method='polynomial')

def prompt_for_elbow():
    elb = input("Enter elbow: ")
    return float(elb)

def plot_all_elbows(x, y, elbows, fig_name):
    kl = plot_kdist(x, y)
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(6, 6))
    plt.title("Elbows")
    plt.plot(kl.x, kl.y, "b", label="data")
    colors = ['r', 'g', 'k', 'm', 'c', 'orange']
    for k, c, s in zip(elbows, colors, elbows):
        plt.hlines(k, plt.xlim()[0], plt.xlim()[1], linestyles='--', colors=c, label=f'eps = {s}')
    plt.legend(loc="best")
    fig.savefig(fig_name, dpi=fig.dpi)

def locate_elbow(distance, k, multiple, auto_detection=True, save_elbow=True, fig_name=""):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='precomputed').fit(distance)
    distances, indices = nbrs.kneighbors(distance)
    kth_dist = distances[:,-1]
    kth_dist.sort()
    x_data = np.arange(len(kth_dist))
    y_data = kth_dist

    elbows = []

    if not multiple:
        kneedle = plot_kdist(x_data, y_data)
        if auto_detection:
            elb = kneedle.elbow_y
        else:
            kneedle.plot_knee()
            plt.show()
            elb = prompt_for_elbow()
        if valid_elbow(elb, y_data[-1]):
            elbows.append(elb)
    else:     
        x_start = 0
        # Min num of points to form an elbow = 3
        if auto_detection:
            while x_start != None and x_start < len(x_data) - 3:
                x = x_data[x_start:]
                y = y_data[x_start:]
                kneedle = plot_kdist(x, y)
                elb = kneedle.elbow_y
                if valid_elbow(elb, y_data[-1]):
                    if elbows and (elb == elbows[-1] or elb / elbows[-1] < 2):
                        # If this elbow is less than double the previous elbow, take the later elbow
                        elbows[-1] = elb
                    else:
                        elbows.append(elb)
                if kneedle.elbow == x_start:
                    break
                x_start = kneedle.elbow
        else:
            kneedle = plot_kdist(x_data, y_data)
            end = False
            while not end:
                kneedle.plot_knee()     
                plt.show()
                elb = prompt_for_elbow()
                if elb == 0:
                    end = True
                if valid_elbow(elb, y_data[-1]):
                    elbows.append(elb)
            
    if save_elbow:
        plot_all_elbows(x_data, y_data, elbows, fig_name)

    return elbows