import numpy as np

def fractionOfEdges(G, communities):
    total_edges = len(G.edges())
    community_labels = sorted(communities)
    label_to_index = {label: idx for idx, label in enumerate(community_labels)}
    n = len(communities)
    e = np.zeros((n, n))

    for u, v in G.edges():
        comm_u = label_to_index[G.nodes[u]['communities']]
        comm_v = label_to_index[G.nodes[v]['communities']]
        if comm_u == comm_v:
            e[comm_u][comm_u] += 1
        else:
            e[comm_u][comm_v] += 1
            e[comm_v][comm_u] += 1

    return e / total_edges, label_to_index


def maxIncMod(e, communities, a):
    n = len(communities)
    best = float('-inf')
    com_i = 0
    com_j = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                k = 2 * (e[i][j] - a[i] * a[j])
                if best < k:
                    best = k
                    com_i = i
                    com_j = j
    return best, com_i, com_j


def fastNewmanAlgorithm(G, nCommunities=2):
    q = 0
    for node in G.nodes():
        G.nodes[node]['communities'] = node
    communities = set([data['communities'] for node, data in G.nodes(data=True)])
    e, label_to_index = fractionOfEdges(G, communities)
    while len(communities) > nCommunities:
        n = e.shape[0]
        a = np.zeros(e.shape[0])
        for i in range(len(a)):
            a[i] = sum(e[i])
        tmp = np.zeros(n)
        for i in range(n):
            tmp[i] = e[i][i] - (a[i] ** 2)

        q = sum(tmp)
        twoDelta, i, j = maxIncMod(e, communities, a)

        index_to_label = {idx: label for label, idx in label_to_index.items()}
        label_i = index_to_label[i]
        label_j = index_to_label[j]

        for node in G.nodes():
            if G.nodes[node]['communities'] == label_j:
                G.nodes[node]['communities'] = label_i

        communities = set([data['communities'] for node, data in G.nodes(data=True)])
        e, label_to_index = fractionOfEdges(G, communities)
    return q