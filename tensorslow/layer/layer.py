from ..core import *
from ..ops import *

# Fully Connected Layer
def fc(input, input_size, size, activation):
    """
    :param input: input vector (input_size x 1)
    :param input_size: dimension of the input vector
    :param size: # of neurons (# of output) => dimension of the output vector
    :param activation: activation function type
    :return: output vector
    """
    # weights (size x input_size)
    weights = Variable((size, input_size), init=True, trainable=True)
    # bias (size x 1)
    bias = Variable((size, 1), init=True, trainable=True)
    affine = Add(MatMul(weights, input), bias)

    if activation == "ReLU":
        return ReLU(affine)
    elif activation == "Logistic":
        return Logistic(affine)
    else:
        return affine

