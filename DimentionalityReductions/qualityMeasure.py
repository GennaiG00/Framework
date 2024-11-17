from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import numpy as np

def class_preservation(data, algorithm_name=''):
    k = 50
    max_distance = 0.2
    nn_model = NearestNeighbors(n_neighbors=k, radius=max_distance)
    nn_model.fit(data.iloc[:, :2])
    distances, indices = nn_model.radius_neighbors(data.iloc[:, :2], return_distance=True)

    preservation_scores = []
    for i, neighbors in enumerate(indices):
        class_i = data.iloc[i]['labels']
        same_class_count = sum(data.iloc[neighbor]['labels'] == class_i for neighbor in neighbors[1:k + 1])
        preservation_score = same_class_count / k
        preservation_scores.append(preservation_score)

    fig, ax = plt.subplots(figsize=(14, 10))
    cmap = plt.get_cmap('hot')
    sc = ax.scatter(data.iloc[:, 0], data.iloc[:, 1], c=preservation_scores, cmap=cmap, edgecolor='k')
    plt.colorbar(sc, ax=ax)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title("Class Preservation Score for " + algorithm_name)

    plt.show()

def evaluate_sammon_mapping(original_data, reduced_data):
    original_distances = pairwise_distances(original_data)
    reduced_distances = pairwise_distances(reduced_data)

    original_distances[original_distances == 0] = 1e-10

    num = np.sum(((original_distances - reduced_distances) ** 2) / original_distances)
    denom = np.sum(original_distances)
    stress = num / denom

    return stress
