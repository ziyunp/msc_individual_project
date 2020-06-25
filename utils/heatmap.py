import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def main():
  for i in range(1, 10):
    # Get data
    labels_file = "labels_" + str(i) + ".csv"
    distance_file = "distance_matrix_" + str(i) + ".csv"
    labels = np.loadtxt(labels_file, delimiter=",")
    distance_matrix = np.loadtxt(distance_file, delimiter=",")
    # Rearrange matrix to order legs by cluster
    ordered_indices = np.argsort(labels)
    length = labels.shape[0]
    order_rows = np.tile(ordered_indices, (length, 1))
    ordered_matrix = np.array(list(map(lambda x, y: y[x], order_rows, distance_matrix)))
    ordered_matrix = ordered_matrix[ordered_indices]
    # Plot heat_map
    heat_map = plt.imshow(ordered_matrix, cmap="hot", interpolation="nearest")
    # Label with clusters
    ordered_labels = labels[ordered_indices]
    label_loc = []
    unique_labels = []
    for i in range(len(ordered_labels)):
      if i == 0:
        label_loc.append(i)
        unique_labels.append(ordered_labels[i])
      else:
        if ordered_labels[i - 1] != ordered_labels[i]:
          label_loc.append(i)
          unique_labels.append(ordered_labels[i])
    plt.xticks(label_loc, unique_labels)
    plt.yticks(label_loc, unique_labels)
    plt.show()

if __name__ == "__main__":
  main()
