from sklearn.metrics import silhouette_score
from sklearn.metrics import jaccard_score
from sklearn.datasets import make_blobs

def silhouette(data, labels):
    return silhouette_score(data, labels)

def jaccard_similarity(set1, set2, average='macro'):
    return jaccard_score(set1, set2, average=average)
