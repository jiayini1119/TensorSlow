# Apply the sobel filters to an image

import sys
sys.path.append('../tensorslow')

import tensorslow as ts
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

pic = matplotlib.image.imread('data/mondrian.jpg') / 255

w, h = pic.shape

# vertical sobel filter
sobel_v = ts.core.Variable(dim=(3, 3), init=False, trainable=False)
sobel_v.set_value(np.mat([[1, 0, -1], [2, 0, -2], [1, 0, -1]]))

# horizontal 
sobel_h = ts.core.Variable(dim=(3, 3), init=False, trainable=False)
sobel_h.set_value(sobel_v.value.T)

img = ts.core.Variable(dim=(w, h), init=False, trainable=False)
img.set_value(np.mat(pic))

# output after applying the filters
sobel_v_output = ts.ops.Convolve(img, sobel_v)
sobel_h_output = ts.ops.Convolve(img, sobel_h)
square_output = ts.ops.Add(
            ts.ops.Multiply(sobel_v_output, sobel_v_output),
            ts.ops.Multiply(sobel_h_output, sobel_h_output)
            )

square_output.forward()

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(221)
ax.axis("off")
ax.imshow(img.value, cmap="gray")

ax = fig.add_subplot(222)
ax.axis("off")
ax.imshow(square_output.value, cmap="gray")

ax = fig.add_subplot(223)
ax.axis("off")
ax.imshow(sobel_v_output.value, cmap="gray")

ax = fig.add_subplot(224)
ax.axis("off")
ax.imshow(sobel_h_output.value, cmap="gray")

plt.show()