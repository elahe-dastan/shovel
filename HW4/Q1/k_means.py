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


def elbow(data, max_k: int):
    clustering_errors = np.zeros(max_k)
    for k in range(1, max_k + 1):
        obj = kmeans(data.to_numpy(), k, 20)
        clusters, centers = obj.cluster()
        clustering_error = obj.clustering_error(clusters, centers)
        clustering_errors[k - 1] = clustering_error

    plt.plot(np.array(range(1, max_k + 1)), clustering_errors)
    plt.title("clustering error for different number of clusters")
    plt.xlabel("number of clusters")
    plt.ylabel("clustering error")
    plt.show()


class kmeans:
    def __init__(self, data: np.ndarray, k: int, iteration: int):
        self.data = data
        self.k = k  # number of clusters
        self.iteration = iteration
        self.n = data.shape[0]  # number of data
        self.c = data.shape[1]  # number of features in the data
        self.colors = ['blue', 'green', 'orange', 'purple']

    def plus_plus(self):
        # Create cluster centroids using the k-means++ algorithm.
        centroids = [self.data[0]]

        for _ in range(1, self.k):
            dist_sq = np.array([min([np.inner(c - x, c - x) for c in centroids]) for x in self.data])
            probs = dist_sq / dist_sq.sum()
            cumulative_probs = probs.cumsum()
            r = np.random.rand()

            i = 0
            for j, p in enumerate(cumulative_probs):
                if r < p:
                    i = j
                    break

            centroids.append(self.data[i])

        return np.array(centroids)

    # It sometimes leads to an error because the cluster of an initial center may become empty
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
            # print("cluster error for cluster {} is {}".format(self.colors[i], m))

        clustering_error = np.mean(cluster_errors)
        print("clustering error is {}".format(clustering_error))
        return clustering_error

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

    def cluster(self) -> [np.ndarray, np.ndarray]:
        centers = self.plus_plus()
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

# data1 = pd.read_csv("Dataset1.csv")
# elbow(data1, 14)
# obj = kmeans(data1.to_numpy(), 4, 20)
# clusters, centers = obj.cluster()
# obj1.show(clusters1, centers1)

data2 = pd.read_csv("Dataset2.csv")
obj = kmeans(data2.to_numpy(), 3, 20)
clusters, centers = obj.cluster()
obj.show(clusters, centers)
