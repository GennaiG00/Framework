from sklearn.metrics import silhouette_score
from sklearn.metrics import jaccard_score
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np

def silhouette(data, labels):
    return silhouette_score(data, labels)

def jaccard_similarity(set1, set2, average='macro'):
    return jaccard_score(set1, set2, average=average)


def cluster_preservation(data, cluster_labels, k=50):
    data_array = np.array(data.iloc[:, :2].values)
    nn_model = NearestNeighbors(n_neighbors=k)
    nn_model.fit(data_array)
    distances, indices = nn_model.kneighbors(data_array)

    fig, ax = plt.subplots(figsize=(10, 10))

    preservation_scores = []
    for i, neighbors in enumerate(indices):
        cluster_i = cluster_labels.iloc[i, 0]
        same_cluster_count = sum(cluster_labels.iloc[neighbor, 0] == cluster_i for neighbor in neighbors[1:])
        preservation_score = same_cluster_count / k
        preservation_scores.append(preservation_score)

    cmap = plt.get_cmap('hot')
    sc = ax.scatter(data.iloc[:, 0], data.iloc[:, 1], c=preservation_scores, cmap=cmap, edgecolor='k')
    plt.colorbar(sc, ax=ax)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title("Cluster Preservation Score")

    plt.show()