import numpy as np


class Node(object):
    """
    abstract base class
    """

    def __init__(self, *parents, **kwargs):
        self.parents = list(parents)
        self.children = []
        self.value = None  # matrix => value.shape => tuple (height, width)
        self.jacobi = None  # jacobian of the result node to itself

        # add itself to its parents' child node
        for parent in self.parents:
            parent.children.append(self)

        self.graph = kwargs.get('graph', default_graph)
        self.need_save = kwargs.get('need_save', True)
        self.gen_node_name(**kwargs)

        def get_parents(self):
            return self.parents

        def get_children(self):
            return self.children

        def gen_node_name(self, **kwargs):
            self.name = kwargs.get('name', '{}:{}'.format(
                self.__class__.__name__, self.graph.node_count()))
            if self.graph.name_scope:
                self.name = '{}/{}'.format(self.graph.name_scope, self.name)

        def compute(self):
            """
            Abstract method - computation depends on the node
            """

        def get_jacobi(self, parent):
            """
            Abstract method - computation depends on the node
            jacobian of child to parent 
            """

        def shape(self):
            return self.value.shape

        def dimension(self):
            return self.value.shape[0] * self.value.shape[1]

        def forward(self):
            """
            For calculating the value for each node, especially the prediction
            """
            for node in self.parents:
                if node.value is None:
                    node.forward()  # Use forward propagation to calculate the value for the parent node
            self.compute()

        def backward(self, result):
            """
            For calculating the jacobian for each node, especially the variable nodes we want to train
            result: result node
            """
            if self.jacobi is None:
                # if itself is the result node, its jacobian to itself is an identity matrix with its dimension
                if self is result:
                    self.jacobi = np.mat(np.eye(self.dimension()))
                else:
                    self.jacobi = np.mat(
                        np.zeros((result.dimension(), self.dimension())))  # accumulator
                    for child in self.get_children():
                        if child.value is not None:
                            self.jacobi += child.get_jacobi(self) * child.backward(result)
            return self.jacobi

        def clear_jacobi(self):
            self.jacobi = None

        def reset_value(self, recursive=True):
            
            self.value = None

            if recursive:
                for child in self.children:
                    child.reset_value()
            
            


        