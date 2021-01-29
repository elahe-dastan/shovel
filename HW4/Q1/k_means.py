from copy import deepcopy

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def scatter(dataset: str):
    data = pd.read_csv(dataset)
    # more generally I should get the header and ...
    x = data["X"]
    y = data["Y"]
    plt.scatter(x, y)
    plt.title("{} Scatter Plot".format(dataset))
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


class kmeans:
    def __init__(self, data: np.ndarray, k: int, iteration: int):
        self.data = data
        self.k = k  # number of clusters
        self.iteration = iteration
        self.n = data.shape[0]  # number of data
        self.c = data.shape[1]  # number of features in the data
        self.colors = ['blue', 'green', 'orange', 'purple']

    def initial_centers(self):
        # Generate random centers using Gaussian distribution
        mean = np.mean(self.data, axis=0)
        std = np.std(self.data, axis=0)
        centers = np.random.randn(self.k, self.c) * std + mean
        return centers

        # plt.scatter(data[:, 0], data[:, 1], s=7)
        # plt.scatter(centers[:, 0], centers[:, 1], marker='*', c='g', s=150)
        # plt.show()

    def clustering_error(self, final_clusters, final_centers):
        cluster_errors = np.zeros(self.k)
        for i in range(self.k):
            norm = np.linalg.norm(self.data[final_clusters == i] - final_centers[i], axis=1)
            m = np.mean(norm, axis=0)
            cluster_errors[i] = m
            print("cluster error for cluster {} is {}".format(self.colors[i], m))

        print("clustering error is {}".format(np.mean(cluster_errors)))

    def show(self, final_clusters, final_centers):
        # plot data points
        for i in range(self.n):
            plt.scatter(self.data[i, 0], self.data[i, 1], s=7, color=self.colors[final_clusters[i]])
        # plot centers
        for i in range(self.k):
            plt.scatter(final_centers[i, 0], final_centers[i, 1], marker='*', c=self.colors[i], s=150)
        plt.title("K-Means on Dataset, rounds={}, k={}".format(self.iteration, self.k))
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

    def cluster(self):
        centers = self.initial_centers()
        centers_new = deepcopy(centers)

        clusters = np.zeros(self.n)

        distances = np.zeros((self.n, self.k))

        for j in range(self.iteration):
            for i in range(self.k):
                distances[:, i] = np.linalg.norm(self.data - centers[i], axis=1)
            clusters = np.argmin(distances, axis=1)

            for i in range(self.k):
                centers_new[i] = np.mean(self.data[clusters == i], axis=0)

        return clusters, centers_new


# scatter("Dataset1.csv")
# scatter("Dataset2.csv")

data1 = pd.read_csv("Dataset1.csv")
obj1 = kmeans(data1.to_numpy(), 2, 20)
clusters1, centers1 = obj1.cluster()
obj1.show(clusters1, centers1)

# print(data1.to_numpy())
# kmeans(data1.to_numpy(), 4)
