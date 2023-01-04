import numpy as np
from ..core import Node


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
