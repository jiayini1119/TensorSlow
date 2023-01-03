class Graph:

    def __init__(self):
        self.nodes = []
        self.name_scope = None

    def add_node(self, node):
        self.nodes.append(node)

    def clear_jacobi(self):
        for node in self.nodes:
            node.clear_jacobi()

    def reset_value(self):
        for node in self.nodes:
            node.reset_value(False)

    def node_count(self):
        return len(self.nodes)


default_graph = Graph()
