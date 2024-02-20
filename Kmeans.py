# To prevent Subscript for class "list" will generate runtime exception; enclose type annotation in quotes
from __future__ import annotations
import seaborn as sns
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler # vector based

def minkowski_dis(x_1, x_2, p):
    from scipy.spatial import distance
    return distance.minkowski(x_1, x_2, p)


class Kmeans:
    def __init__(self, K, itterations, p, random_state=15):
        self.K = K
        self.itterations = itterations
        self.p = p
        # self.clusters = [[] for _ in range(self.K)]
        self.clusters = []
        for _ in range(self.K):
            self.clusters.append([])
        self.centers = []
        self.inertia = 0.0
        self.random_state = random_state
        np.random.seed(random_state)

    # def k_plus_plus(self, datapoints: np.array, k, random_state=94):
    #     np.random.seed(random_state)
    #     centers = [datapoints[0]]

    #     for _ in range(1, k):
    #         random = np.random.rand()
    #         distance_squared = np.array([min([np.inner(c-x, c-x) for c in centers]) for x in datapoints])
    #         probabilities = distance_squared / distance_squared.sum()
    #         cumulative_probabilities = probabilities.cumsum()

    #         for i, j in enumerate(cumulative_probabilities):
    #             if random < j:
    #                 center = i
    #                 break

    #         centers.append(datapoints[center])

    #     return np.array(centers)

    def predict(self, X, flag=0):
        self.X = X
        self.num_samples, self.num_features = self.X.shape

        if flag == 0:
            np.random.seed(self.random_state)
            centers = np.zeros((self.K, self.num_features))
            for k in range(self.K):
                centroid = X[np.random.choice(range(self.num_samples))]
                centers[k] = centroid
            self.centers = centers

        elif flag == 1:
            # self.centers = self.k_plus_plus(X, self.K)
            lst_centers = [X[0]]
            for _ in range(1, self.K):
                random = np.random.rand()
                distance_squared = np.array([min([np.inner(c-x, c-x) for c in lst_centers]) for x in X])
                probabilities = distance_squared / distance_squared.sum()
                cumulative_probabilities = probabilities.cumsum()
                for i, j in enumerate(cumulative_probabilities):
                    if random < j:
                        center = i
                        break

                lst_centers.append(X[center])
            self.centers = np.array(lst_centers)

        # elif flag == 2:
        #     self.centers = self.naive_sharding(X, self.K)

        for _ in range(self.itterations):
            # self.clusters = self.cluster_creation(self.centers)
            self.clusters = [[] for _ in range(self.K)]
            for idx, sample in enumerate(self.X):
                # centroid_idx = self.nearby_centroid(sample, self.centers)
                distances = [minkowski_dis(sample, point, self.p) for point in centers]
                centroid_idx = np.argmin(distances)
                self.clusters[centroid_idx].append(idx)
            centroids_old = self.centers
            # self.centers = self.get_centroids(self.clusters)
            self.centers = np.zeros((self.K, self.num_features))
            for cluster_idx, cluster in enumerate(self.clusters):
                if (len(cluster) != 0):
                    cluster_mean = np.mean(self.X[cluster], axis=0)
                    self.centers[cluster_idx] = cluster_mean
            # if self.convergence_test(centroids_old, self.centers):
            distances = [minkowski_dis(centroids_old[i], self.centers[i], self.p) for i in range(self.K)]
            if sum(distances) == 0:
                break
        # labels = self.cluster_labels(self.clusters)
        labels = np.empty(self.num_samples)
        for cluster_idx, cluster in enumerate(self.clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        # self.calculateInertia(X, labels)
        labels = labels.astype(int)
        for idx, pts in enumerate(X):
            self.inertia += minkowski_dis(
                self.centers[labels[idx]], pts, 2)**2
        return labels, self.centers

    # def cluster_creation(self, centers):
    #     clusters = [[] for _ in range(self.K)]
    #     for idx, sample in enumerate(self.X):
    #         centroid_idx = self.nearby_centroid(sample, centers)
    #         clusters[centroid_idx].append(idx)
    #     return clusters

    # def nearby_centroid(self, sample, centers):
    #     # distance metric
    #     distances = [minkowski_dis(sample, point, self.p) for point in centers]
    #     closest_idx = np.argmin(distances)
    #     return closest_idx

    # def get_centroids(self, clusters):
    #     centers = np.zeros((self.K, self.num_features))
    #     for cluster_idx, cluster in enumerate(clusters):
    #         if (len(cluster) != 0):
    #             cluster_mean = np.mean(self.X[cluster], axis=0)
    #             centers[cluster_idx] = cluster_mean
    #     return centers

    # def convergence_test(self, centroids_old, centers):
    #     distances = [minkowski_dis(centroids_old[i], centers[i], self.p) for i in range(self.K)]
    #     return sum(distances) == 0

    # def cluster_labels(self, clusters):
    #     labels = np.empty(self.num_samples)
    #     for cluster_idx, cluster in enumerate(clusters):
    #         for sample_idx in cluster:
    #             labels[sample_idx] = cluster_idx
    #     return labels

    # def calculateInertia(self, datapoints: list[int], labels: list[any]) -> None:
    #     labels = labels.astype(int)
    #     for idx, pts in enumerate(datapoints):
    #         self.inertia += minkowski_dis(
    #             self.centers[labels[idx]], pts, 2)**2

    def predictPoint(self, datapoint):
        distances = []
        for idx, i in enumerate(self.centers):
            if sum(i) == 0:
                continue
            else:
                distances.append([minkowski_dis(datapoint, i, self.p), idx])
        distances.sort()
        return distances[0][1]
