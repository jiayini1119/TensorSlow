# Train a random filter into a sobel filter

import sys
sys.path.append('../tensorslow')

import tensorslow as ts
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

pic = matplotlib.image.imread('data/lena.jpg') / 255

w, h = pic.shape

# sobel
sobel = ts.core.Variable(dim=(3, 3), init=False, trainable=False)
sobel.set_value(np.mat([[1, 0, -1], [2, 0, -2], [1, 0, -1]]))

img = ts.core.Variable(dim=(w, h), init=False, trainable=False)
img.set_value(np.mat(pic))

sobel_output = ts.ops.Convolve(img, sobel)

sobel_output.forward()
plt.imshow(sobel_output.value, cmap="gray")

# filter to be trained
filter_train = ts.core.Variable(dim=(3, 3), init=True, trainable=True)
filter_output = ts.ops.Convolve(img, filter_train)

# for matrix subtraction
minus = ts.core.Variable(dim=(w, h), init=False, trainable=False)
minus.set_value(np.mat(-np.ones((w, h))))

# mean square error
error = ts.ops.Add(sobel_output, ts.ops.Multiply(filter_output, minus))
square_error = ts.ops.MatMul(ts.ops.Reshape(error, shape=(1, w * h)), ts.ops.Reshape(error, shape=(w * h, 1)))
n = ts.core.Variable((1, 1), init=False, trainable=False)
n.set_value(np.mat(1.0 / (w * h)))
mse = ts.ops.MatMul(square_error, n)

optimizer = ts.optimizer.Adam(ts.core.default_graph, mse, 0.01)

# training
for i in range(1000):
    optimizer.one_step()
    optimizer.update()
    mse.forward()
    print("iteration:{:d},loss:{:.10f}".format(i, mse.value[0, 0]))

filter_train.forward()
print(filter_train.value) # should be very close to the value of the sobel

filter_output.forward()
plt.imshow(filter_output.value, cmap="gray")