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
