import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def main():
  for i in range(1, 10):
    # Group legs from the same cluster together,

    distance_file = "distance_matrix_" + str(i) + ".csv"
    distance_matrix = np.loadtxt(distance_file, delimiter=",")
    heat_map = plt.imshow(distance_matrix, cmap="hot", interpolation="nearest")
    plt.show()

if __name__ == "__main__":
  main()
