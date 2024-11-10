import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx

def plot2D(data, labels):
    if isinstance(labels, pd.DataFrame):
        labels = labels.iloc[:, 0]
    fig, ax = plt.subplots(figsize=(15, 15))
    points = np.array(data)
    unique_labels = set(labels)
    colors = plt.get_cmap('tab20')(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        label_points = points[np.array(labels) == label]
        if label == -1:
            ax.scatter(label_points[:, 0], label_points[:, 1], color='black', label='Noise', marker='x')
        else:
            ax.scatter(label_points[:, 0], label_points[:, 1], color=color, label=f'Cluster {label}')

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_communities(G):
    communities = [G.nodes[node]['communities'] for node in G.nodes()]
    uniqueCommunities = list(set(communities))
    color_map = {community: idx for idx, community in enumerate(uniqueCommunities)}
    node_colors = [color_map[G.nodes[node]['communities']] for node in G.nodes()]
    pos = nx.spring_layout(G)
    plt.figure(figsize=(10, 7))
    nx.draw(G, pos, node_color=node_colors, node_size=500, cmap=plt.cm.jet)
    plt.title("Graph with detected communities")
    plt.show()

def print_communities(G):
    communities = [G.nodes[node]['communities'] for node in G.nodes()]
    uniqueCommunities = list(set(communities))
    for comm in uniqueCommunities:
        print("(", end='')
        for node in G.nodes():
            if G.nodes[node]['communities'] == comm:
                print(node, end='')
                if node != G.nodes(-1) :
                    print(",", end='')
        print(")")


def print_communities_GN(G, communities):
    color_map = []
    for node in G:
        for i, community in enumerate(communities):
            if node in community:
                color_map.append(i)

    nx.draw(G, node_color=color_map, with_labels=False, cmap=plt.cm.tab10)
    plt.show()


