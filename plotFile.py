import matplotlib.pyplot as plt
import numpy as np
def plotClusters(points):
    # Estrai x, y e i label dai risultati
    x = points[:, 0]  # Prima colonna per le x
    y = points[:, 1]  # Seconda colonna per le y
    labels = points[:, 2]  # Terza colonna per i label

    # Determina i cluster unici (label)
    unique_labels = np.unique(labels)

    # Crea una mappa di colori per i cluster
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    plt.figure(figsize=(10, 8))

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Rumore (punti con label -1)
            col = [0, 0, 0, 1]  # Colore nero per il rumore
            cluster_label = 'Noise'
        else:
            cluster_label = f'Cluster {k}'

        # Maschera per filtrare i punti del cluster k
        class_member_mask = (labels == k)

        # Plotta i punti
        xy = np.vstack((x[class_member_mask], y[class_member_mask])).T
        plt.scatter(xy[:, 0], xy[:, 1], s=30, c=[col], label=cluster_label)

    # Etichette e titolo del grafico
    plt.title('DBSCAN Clustering')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.show()
