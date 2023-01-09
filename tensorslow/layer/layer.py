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

# Convolutional Layer
def conv(feature_maps, input_shape, kernels, kernel_shape, activation):
    """
    :param feature_maps: an array of feature maps
    :param input_shape: size of the feature map
    :param kernels: # kernels == # feature maps == # channels
    :param kernel_shape: shape of the kernel
    :param activation: activation function type
    :return: an array of feature maps, can be sent to another layer or the pool 
    """

    # for constructing the bias
    ones = Variable(input_shape, init=False, trainable=False)
    ones.set_value(np.mat(np.ones(input_shape)))

    outputs = [] # store the feature maps
    for i in range(kernels):
        channels = [] # store the filtered output for every channel
        for fm in feature_maps:
            kernel = Variable(kernel_shape, init=True, trainable=True)
            conv = Convolve(fm, kernel)
            channels.append(conv)
    
    channels = Add(*channels)
    bias = ScalarMultiply(Variable(dim=(1, 1), init=True, trainable=True), ones)
    affine = Add(channels, bias)

    if activation == "ReLU":
        outputs.append(ReLU(affine))
    elif activation == "Logistic":
        outputs.append(Logistic(affine))
    else:
        outputs.append(affine)    
    
    assert len(outputs) == kernels
    return outputs