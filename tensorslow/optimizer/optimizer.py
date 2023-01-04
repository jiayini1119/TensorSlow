import numpy as np

from ..core import Node, Variable, get_node_from_graph
from ..core.graph import Graph


class Optimizer(object):

    def __init__(self, graph, target, learning_rate=0.01):
        assert isinstance(target, Node) and isinstance(graph, Graph)
        self.graph = graph
        self.target = target  # only optimize one target node
        self.learning_rate = learning_rate

        self.acc_gradient = dict()
        self.acc_no = 0  # counter

    def get_gradient(self, node):
        assert node in self.acc_gradient
        return self.acc_gradient[node] / self.acc_no  # average gradient

    def forward_backward(self):
        self.graph.clear_jacobi()
        self.target.forward()

        # look for trainable variable nodes and apply backward propagation
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                node.backward(self.target)
                # gradient is the transpose of the jacobian
                gradient = node.jacobi.T.reshape(node.shape())
                if node not in self.acc_gradient:
                    self.acc_gradient[node] = gradient
                else:
                    self.acc_gradient[node] += gradient

    def one_step(self):
        self.forward_backward()
        self.acc_no += 1

    def _update(self):
        """
        Abstract method
        """
        pass

    def apply_gradients(self, node_gradients_dict, summarize=False, acc_no=None):
        for node, gradient in node_gradients_dict.items():
            if isinstance(node, Node):
                pass
            else:
                target_node = get_node_from_graph(node)
                assert target_node is not None
                assert self.acc_gradient[target_node].shape == gradient.shape
                if summarize:
                    self.acc_gradient[target_node] += gradient
                else:
                    self.acc_gradient[target_node] = gradient

        if summarize:
            self.acc_no += acc_no
        else:
            if acc_no is None:
                self.acc_no = 1
            else:
                self.acc_no = acc_no

    def update(self, var_gradients=None):
        if var_gradients is not None:
            self.apply_gradients(var_gradients)

        self._update()
        self.acc_gradient.clear()
        self.acc_no = 0


class GradientDescent(Optimizer):
    def __init__(self, graph, target, learning_rate=0.01):
        Optimizer.__init__(self, graph, target)
        self.learning_rate = learning_rate

    def _update(self):
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                gradient = self. get_gradient(node)
                node.set_value(node.value - self.learning_rate * gradient)

class Momentum(Optimizer):
    def __init__(self, graph, target, learning_rate=0.01, momentum=0.9):
        Optimizer.__init__(self, graph, target) 
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.v = dict() # previous v

    def _update(self):
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                gradient = self.get_gradient(node)
                
                if node not in self.v:
                    self.v[node] = gradient
                else:
                    self.v[node] = self.momentum * self.v[node] - self.learning_rate * gradient
                
                node.set_value(node.value + self.v[node])

class AdaGrad(Optimizer):
    def __init__(self, graph, target, learning_rate=0.01):
        Optimizer.__init__(self, graph, target)
        self.learning_rate = learning_rate
        self.s = dict()
    
    def _update(self):
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                gradient = self.get_gradient(node)

                if node not in self.s:
                    self.s[node] = np.power(gradient, 2)
                else:
                    self.s[node] = self.s[node] + np.power(gradient, 2)

                node.set_value(node.value - self.learning_rate * gradient / (np.sqrt(self.s[node] + 1e-10)))


class RMSProp(Optimizer):
    def __init__(self, graph, target, learning_rate=0.01, beta=0.9):
        Optimizer.__init__(self, graph, target)
        self.learning_rate = learning_rate
        assert 0.0 < beta < 1.0
        self.beta = beta
        self.s = dict()

    def _update(self):
        for node in self.graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                gradient = self.get_gradient(node)

                if node not in self.s:
                    self.s[node] = np.power(gradient, 2)
                else:
                    self.s[node] = self.beta * self.s[node] + (1 - self.beta) * np.power(gradient, 2)

                node.set_value(node.value - self.learning_rate *
                               gradient / (np.sqrt(self.s[node] + 1e-10)))
