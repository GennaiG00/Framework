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

    return sum(preservation_scores) / len(preservation_scores)

def evaluate_sammon_mapping(original_data, reduced_data):
    original_distances = pairwise_distances(original_data)
    reduced_distances = pairwise_distances(reduced_data)

    original_distances[original_distances == 0] = 1e-10

    num = np.sum(((original_distances - reduced_distances) ** 2) / original_distances)
    denom = np.sum(original_distances)
    stress = num / denom

    return stress


def trustworthiness(X, X_reduced, n_neighbors=20):
    # Verifica che n_neighbors sia valido rispetto al numero di osservazioni
    n_neighbors = min(n_neighbors, len(X))

    # Costruisci NearestNeighbors sui dati originali e ridotti
    nn_original = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
    nn_reduced = NearestNeighbors(n_neighbors=n_neighbors).fit(X_reduced)

    trust = 0
    print(len(X))
    for i in range(len(X)):
        # Trova i vicini nei dati originali
        neighbors_original = nn_original.kneighbors([X[i]], n_neighbors=n_neighbors, return_distance=False)
        # Trova i vicini nei dati ridotti
        neighbors_reduced = nn_reduced.kneighbors([X_reduced[i]], n_neighbors=n_neighbors, return_distance=False)

        # Confronta i vicini trovati
        trust += len(set(neighbors_original[0]).intersection(set(neighbors_reduced[0])))

    # Calcola la fidatezza normalizzata
    return trust / (len(X) * n_neighbors)



def continuity(X, X_reduced, n_neighbors=20):
    # Verifica che n_neighbors sia valido rispetto al numero di osservazioni
    n_neighbors = min(n_neighbors, len(X))

    # Costruisci NearestNeighbors sui dati originali e ridotti
    nn_original = NearestNeighbors(n_neighbors=n_neighbors).fit(X)
    nn_reduced = NearestNeighbors(n_neighbors=n_neighbors).fit(X_reduced)

    continuity_score = 0
    for i in range(len(X)):
        # Trova i vicini nei dati originali
        neighbors_original = nn_original.kneighbors([X[i]], n_neighbors=n_neighbors, return_distance=False)[0]
        # Trova i vicini nei dati ridotti
        neighbors_reduced = nn_reduced.kneighbors([X_reduced[i]], n_neighbors=n_neighbors, return_distance=False)[0]

        # Individua i vicini persi (presenti in X ma non in X_reduced)
        lost_neighbors = set(neighbors_original) - set(neighbors_reduced)

        # Incrementa lo score basandosi su quanti persi sono stati effettivamente trovati
        continuity_score += len(lost_neighbors)

    # Normalizza il punteggio
    total_possible_lost = len(X) * n_neighbors
    return 1 - (continuity_score / total_possible_lost)
