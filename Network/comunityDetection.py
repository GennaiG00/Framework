from networkx.algorithms.community import girvan_newman
from Network.FastNewman import *
import networkx as nx
import matplotlib.pyplot as plt

class CommunityDetection:
    def __init__(self, detection_algorithm):
        self.detection_algorithm = None
        if detection_algorithm == "fastNewman":
            self.detection_algorithm = "fastNewman"
        elif detection_algorithm == "girvanNewman":
            self.detection_algorithm = "girvanNewman"

    def detection(self, network, nCommunities=2):
        if self.detection_algorithm == "fastNewman":
            return fastNewmanAlgorithm(network, nCommunities=nCommunities)
        elif self.detection_algorithm == "girvanNewman":
            return girvan_newman(network)

    def plot_communities(G, communities, method, file_name):
        num_communities = len(communities)
        color_map = plt.get_cmap('tab20')
        colors = [color_map(i / max(num_communities - 1, 1)) for i in range(num_communities)]

        node_colors = {}
        for i, community in enumerate(communities):
            for node in community:
                node_colors[node] = colors[i]

        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True,
                node_color=[node_colors.get(node, 'gray') for node in G.nodes()],
                node_size=500, edge_color='gray', font_size=10)
        plt.title(f"{method} - Communities in {file_name}")
        plt.show()
