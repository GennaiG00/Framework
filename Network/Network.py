import networkx as nx
import matplotlib.pyplot as plt

class netoworkWithNx:
    def __init__(self, graph):
        self.graph = graph

    def getGraph(self):
        return self.graph

    def plotGraph(self):
        nx.draw(self.graph, with_labels=False, node_color='red')
        plt.show()

    def getNodes(self):
        return self.graph.nodes()

    def getEdges(self):
        return self.graph.edges()

    def addNode(self, node):
        self.graph.add_node(node)

    def addEdge(self, node1, node2):
        self.graph.add_edge(node1, node2)

    def computePageRank(self, alpha=0.85):
        pagerank = nx.pagerank(self.graph, alpha=alpha)
        return pagerank

    def edgeBetweeness(self):
        return nx.edge_betweenness_centrality(self.graph, normalized=True)