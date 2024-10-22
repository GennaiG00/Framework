

def evaluate_dimensionality_reduction(pca_model):
    explained_variance = pca_model.explained_variance_ratio_
    total_explained_variance = explained_variance.sum()
    return total_explained_variance * 100

