import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import config

def arrange_matrix_by_cluster(matrix, labels):
  ordered_indices = np.argsort(labels)
  length = labels.shape[0]
  order_rows = np.tile(ordered_indices, (length, 1))
  ordered_matrix = np.array(list(map(lambda x, y: y[x], order_rows, matrix)))
  return ordered_matrix[ordered_indices], ordered_indices

def main():
  for i in range(1, 10):
    # Get data
    labels_file = config.DATA_FOR_HEATMAP["labels"] + str(i) + ".csv"
    distance_file = config.DATA_FOR_HEATMAP["distances"] + str(i) + ".csv"
    labels = np.loadtxt(labels_file, delimiter=",")
    distance_matrix = np.loadtxt(distance_file, delimiter=",")
    
    # Rearrange matrix to order legs by cluster
    ordered_matrix, ordered_indices = arrange_matrix_by_cluster(distance_matrix, labels)
    
    # Save ordered matrix to file
    matrix_file = "ordered_matrix_" + str(i) + ".csv"
    indices_file = "ordered_indices_" + str(i) + ".csv"
    np.savetxt(matrix_file, ordered_matrix, delimiter=",")
    np.savetxt(indices_file, ordered_indices, delimiter=",")

    # Plot heat_map
    fig = plt.figure(figsize=(6, 6))
    fig_name = "Heatmap_" + str(i) + ".png"
    heat_map = plt.imshow(ordered_matrix, cmap="hot")

    # Label heatmap with cluster labels
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
    fig.savefig(fig_name, dpi=fig.dpi)

if __name__ == "__main__":
  main()
