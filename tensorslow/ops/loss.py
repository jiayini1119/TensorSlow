import numpy as np
from ..core import Node
from ..ops import SoftMax


class LossFunction(Node):
    pass


class PerceptionLoss(LossFunction):
    def compute(self):
        self.value = np.mat(
            np.where(self.parents[0].value >= 0.0, 0.0, -self.parents[0].value))

    def get_jacobi(self, parent):
        diag = np.where(parent.value >= 0.0, 0.0, -1)
        return np.diag(diag.ravel())


class LogLoss(LossFunction):
    def compute(self):
        assert len(self.parents) == 1
        x = self.parents[0].value
        # set limit as 1e2 to avoid overflow
        self.value = np.log(1 + np.power(np.e, np.where(-x > 1e2, 1e2, -x)))
    
    def get_jacobi(self, parent):
        x = parent.value
        diag = -1 / (1 + np.power(np.e, np.where(x > 1e2, 1e2, x)))
        return np.diag(diag.ravel())


class CrossEntropyWithSoftMax(LossFunction):
    # apply SoftMax to parents[0].value
    # parents[1].value = One Hot
    # compare
    def compute(self):
        prob = SoftMax.softmax(self.parents[0].value)
        self.value = np.mat(-np.sum(np.multiply(self.parents[1].value, np.log(prob + 1e-10))))


    def get_jacobi(self, parent):
        prob = SoftMax.softmax(self.parents[0].value)
        if parent is self.parents[0]:
            return (prob - self.parents[1].value).T
        else:
            return (-np.log(prob)).T
