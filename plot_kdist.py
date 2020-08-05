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

def prompt_for_elbow(detected_elbow):
    accept = input("Accept auto-detected elbow: {}? y/n:  ".format(str(detected_elbow)))
    while accept != "n" and accept != "y":
        accept = input("Accept auto-detected elbow: {}? y/n:  ".format(str(detected_elbow)))        
    if accept == "n":
        elb = input("Enter elbow (0 to skip this elbow or -1 to end): ")
        is_float = False
        while not is_float:
            try: 
                elb = float(elb)
                is_float = True
            except:
                elb = input("Elbow must be a float value! Enter elbow (0 to skip this elbow or -1 to end): ")
        return elb
    return detected_elbow

def plot_all_elbows(x, y, elbows, save=False, fig_name=""):
    kl = plot_kdist(x, y)
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(6, 6))
    plt.title("Elbows")
    plt.plot(kl.x, kl.y, "b", label="data")
    colors = ['r', 'g', 'k', 'm', 'c', 'orange']
    for k, c, s in zip(elbows, colors, elbows):
        plt.hlines(k, plt.xlim()[0], plt.xlim()[1], linestyles='--', colors=c, label=f'eps = {s}')
    plt.legend(loc="best")
    if save:
        fig.savefig(fig_name, dpi=fig.dpi)

def locate_elbow(distance, k, multiple=False, save_elbow=False, fig_name=""):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='precomputed').fit(distance)
    distances, indices = nbrs.kneighbors(distance)
    kth_dist = distances[:,-1]
    kth_dist.sort()
    x_data = np.arange(len(kth_dist))
    y_data = kth_dist

    elbows = []

    # if not multiple:
        # kneedle = plot_kdist(x_data, y_data)
        # while not elbows:
        #     plot_all_elbows(x_data, y_data, [kneedle.elbow_y])
        #     plt.show()
        #     elb = prompt_for_elbow(kneedle.elbow_y)
        #     if valid_elbow(elb, y_data[-1]):
        #         elbows.append(elb)
    # else:     
    x_start = 0
    end = False
    while not end:
        x = x_data[x_start:]
        y = y_data[x_start:]
        kneedle = plot_kdist(x, y)
        plot_all_elbows(x_data, y_data, elbows + [kneedle.elbow_y])
        plt.show()
        elb = prompt_for_elbow(kneedle.elbow_y)
        if valid_elbow(elb, y_data[-1]):
            elbows.append(elb)
        if elb == -1 or kneedle.elbow == x_start or (not multiple and len(elbows) == 1):
            end = True
        x_start = kneedle.elbow

    if save_elbow:
        plot_all_elbows(x_data, y_data, elbows, True, fig_name)

    return elbows