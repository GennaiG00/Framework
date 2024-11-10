from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import DimentionalityReductions.SammonMapping as sm

class dimensionalityReduction:
    def __init__(self, n_components=2, *hyperparameters):
        self.hyperparameters = hyperparameters
        self.n_components = n_components

    def reduce(self, data, method):
        if method == 'PCA':
            X_normalized = normalize(data)
            X_normalized = pd.DataFrame(X_normalized)
            pca = PCA(n_components=self.n_components)
            x_principal = pca.fit_transform(X_normalized)
            x_principal = pd.DataFrame(x_principal)
            x_principal.columns = ['P1', 'P2']
            return x_principal
        elif method == 't-SNE':
            X_normalized = normalize(data)
            X_normalized = pd.DataFrame(X_normalized)
            tsne = TSNE(n_components=self.n_components, random_state=42)
            x_principal = tsne.fit_transform(X_normalized)
            x_principal = pd.DataFrame(x_principal)
            x_principal.columns = ['P1', 'P2']
            return x_principal
        elif method == 'sammonMapping':
            X_normalized = normalize(data)
            X_normalized = pd.DataFrame(X_normalized)
            iter =  self.hyperparameters[0]
            mf = self.hyperparameters[1]
            initialization = self.hyperparameters[2]
            y , e = sm.sammonMapping(X_normalized.values, self.n_components, iter, mf, initialization)
            return pd.DataFrame(y, columns=['P1', 'P2']), e
        else:
            raise ValueError("Unsupported method for dimensionality reduction")
