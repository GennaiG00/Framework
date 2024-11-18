import networkx as nx

def modularity(graph, communities):

    m = graph.size(weight='weight')
    modularity_score = 0

    for community in communities:
        subgraph = graph.subgraph(community)
        lc = subgraph.size(weight='weight')
        dc = sum(dict(graph.degree(community, weight='weight')).values())
        modularity_score += (lc / m) - (dc / (2 * m))**2

    return modularity_score

def internal_density(graph, communities):

    densities = []
    for community in communities:
        subgraph = graph.subgraph(community)
        n = len(subgraph.nodes())
        if n > 1:
            density = 2 * subgraph.size() / (n * (n - 1))
            densities.append(density)
        else:
            densities.append(0)

    return sum(densities) / len(densities) if densities else 0


def ratio_cut(graph, communities):

    ratio_cut_score = 0
    total_nodes = len(graph.nodes())

    for community in communities:
        cut_edges = nx.cut_size(graph, community)
        size = len(community)
        if size > 0:
            ratio_cut_score += cut_edges / size

    return ratio_cut_score
