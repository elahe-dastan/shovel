from copy import deepcopy

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def kmeans(data: np.ndarray, k: int):
    # number of data
    n = data.shape[0]
    # number of features in the data
    c = data.shape[1]
    # Generate random centers using Gaussian distribution
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    centers = np.random.randn(k, c) * std + mean

    # plt.scatter(data[:, 0], data[:, 1], s=7)
    # plt.scatter(centers[:, 0], centers[:, 1], marker='*', c='g', s=150)
    # plt.show()

    centers_old = np.zeros(centers.shape)
    centers_new = deepcopy(centers)
    clusters = np.zeros(n)
    distances = np.zeros((n, k))
    # norm 2 with zero
    error = np.linalg.norm(centers_new - centers_old)

    rounds = 20
    for j in range(rounds):
        for i in range(k):
            distances[:, i] = np.linalg.norm(data - centers[i], axis=1)
        clusters = np.argmin(distances, axis=1)

        centers_old = deepcopy(centers_new)
        for i in range(k):
            centers_new[i] = np.mean(data[clusters == i], axis=0)
        # error = np.linalg.norm(centers_new - centers_old)
    print(centers_new)

    colors = ['blue', 'green', 'orange', 'purple']
    # plot data points
    for i in range(n):
        plt.scatter(data[i, 0], data[i, 1], s=7, color=colors[clusters[i]])
    # plot centers
    for i in range(k):
        plt.scatter(centers_new[i, 0], centers_new[i, 1], marker='*', c=colors[i], s=150)
    plt.title("K-Means on Dataset1, rounds={}, k={}".format(rounds, k))
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


# illustrating dataset 1
data1 = pd.read_csv("Dataset1.csv")
# X1 = data1["X"]
# Y1 = data1["Y"]
# print(type(X1))

# plt.scatter(X1, Y1)
# plt.title("Dataset1 Scatter Plot")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.show()

# illustrating dataset 2
# data2 = pd.read_csv("Dataset2.csv")
# X2 = data2["X"]
# Y2 = data2["Y"]
#
# plt.scatter(X2, Y2)
# plt.title("Dataset2 Scatter Plot")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.show()

kmeans(data1.to_numpy(), 4)
