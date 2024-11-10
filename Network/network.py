import networkx as nx

class Network:
    def __init__(self, G):
        self.nodes = {}
        self.edges = []
        self.node_mapping = {}
        self.network = nx.Graph()
        node_ids = [node['id'] for node in nodes]  # Extracts node identifiers
        self.network.add_nodes_from(node_ids)
        edges = [(edge['source'], edge['target']) for edge in edges]  # Adjust keys as needed
        self.network.add_edges_from(edges)

        if nodes and isinstance(nodes, list):
            for i, node in enumerate(nodes):
                node_name = node["name"]
                self.node_mapping[node_name] = i  # Map node name to an integer
                self.add_node( **{k: v for k, v in node.items() if k != "name"})

        if edges and isinstance(edges, list):
            for edge in edges:
                source = edge['source']
                target = edge['target']
                if source is not None and target is not None:
                    self.add_edge(source, target)

    def add_node(self, **attributes):
        self.nodes[len(self.nodes)] = {"edges": [], "attributes": attributes}

    def add_edge(self, node1, node2):
        if node1 in self.nodes and node2 in self.nodes:
            edge = {"nodes": (node1, node2)}
            self.edges.append(edge)
            self.nodes[node1]["edges"].append(node2)
            self.nodes[node2]["edges"].append(node1)
        else:
            raise ValueError("Both nodes must be exist in the network.")

    def display_network(self):
        print("Nodes (with integer indices):")
        for node_id, data in self.nodes.items():
            print(f"  Node {node_id}: {data['attributes']}")

        print("\nEdges (with integer indices):")
        for edge in self.edges:
            print(f"  {edge['nodes']}")

    def check_node(self, nodeToCheck=None):
        if nodeToCheck is None:
            return False
        for node_data in self.nodes.values():
            if node_data["attributes"]["value"] == nodeToCheck:
                return True
        return False