from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import silhouette_score

def evaluate_clustering(data, labels):
    filtered_data = data[labels != -1]
    filtered_labels = labels[labels != -1]

    if len(set(filtered_labels)) > 1:
        score = silhouette_score(filtered_data, filtered_labels)
        return score
    else:
        raise Exception("We need at least 2 clusters to calculate the Silhouette Score.")


def evaluate_clustering_dbi(data, labels):
    filtered_data = data[labels != -1]
    filtered_labels = labels[labels != -1]

    if len(set(filtered_labels)) > 1:
        dbi = davies_bouldin_score(filtered_data, filtered_labels)
        return dbi
    else:
        raise Exception("We need at least 2 clusters to calculate the Davies-Bouldin Index.")
