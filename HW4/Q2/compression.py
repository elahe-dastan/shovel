from copy import deepcopy

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from ..Q1.k_means import kmeans


def kmeans(data: np.ndarray, k: int) -> [np.ndarray, np.ndarray]:
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
    return clusters, centers_new

    # colors = ['blue', 'green', 'orange', 'purple']
    # cluster_errors = np.zeros(k)
    # for i in range(k):
    #     norm = np.linalg.norm(data[clusters == i] - centers[i], axis=1)
    #     m = np.mean(norm, axis=0)
    #     cluster_errors[i] = m
    #     print("cluster error for cluster {} is {}".format(colors[i], m))
    #
    # print("clustering error is {}".format(np.mean(cluster_errors)))
    # # plot data points
    # for i in range(n):
    #     plt.scatter(data[i, 0], data[i, 1], s=7, color=colors[clusters[i]])
    # # plot centers
    # for i in range(k):
    #     plt.scatter(centers_new[i, 0], centers_new[i, 1], marker='*', c=colors[i], s=150)
    # plt.title("K-Means on Dataset1, rounds={}, k={}".format(rounds, k))
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.show()


img = mpimg.imread('sample_img1.png')
# converting image to two dimensional
two_d_image = img.reshape(-1, 3)
clusters, centers = kmeans(two_d_image, 64)

for i in range(two_d_image.shape[0]):
    two_d_image[i] = centers[clusters[i]]

compressed_image = img.reshape(img.shape[0], img.shape[1], 3)
plt.imshow(compressed_image)
plt.show()