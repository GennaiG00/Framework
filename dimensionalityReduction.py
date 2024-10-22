from macholib.mach_o import note_command
from networkx.algorithms.cuts import normalized_cut_size
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import dimentionalityReductions.SammonMapping as sm

from dimentionalityReductions.SammonMapping import sammonMapping


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
            return x_principal, pca
        elif method == 't-SNE':
            tsne = TSNE(n_components=self.n_components, random_state=42)
            x_principal = tsne.fit_transform(data)
            return x_principal
        elif method == 'sammonMapping':
            iter =  self.hyperparameters[0]
            mf = self.hyperparameters[1]
            initialization = self.hyperparameters[2]
            y , e = sm.sammonMapping(data, self.n_components, iter, mf, initialization)
            return y, e
        else:
            raise ValueError("Unsupported method for dimensionality reduction")
