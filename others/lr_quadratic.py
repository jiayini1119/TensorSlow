import sys
sys.path.append('../tensorslow')

import numpy as np
import tensorslow as ts
from sklearn.datasets import make_circles

X, y = make_circles(200, noise=0.1, factor=0.2)
y = y * 2 - 1 

use_quadratic = True

x1 = ts.core.Variable(dim=(2, 1), init=False, trainable=False)
label = ts.core.Variable(dim=(1, 1), init=False, trainable=False)
b = ts.core.Variable(dim=(1, 1), init=True, trainable=True)

if use_quadratic:
    # feature engineering
    x2 = ts.ops.Reshape(ts.ops.MatMul(x1, ts.ops.Reshape(x1, shape=(1, 2))), shape=(4, 1))
    x = ts.ops.Concat(x1, x2)
    w = ts.core.Variable(dim=(1, 6), init=True, trainable=True)

else:
    x = x1
    w = ts.core.Variable(dim=(1, 2), init=True, trainable=True)

output = ts.ops.Add(ts.ops.MatMul(w, x), b)
predict = ts.ops.Logistic(output)
loss = ts.ops.loss.LogLoss(ts.ops.Multiply(label, output))
learning_rate = 0.001

optimizer = ts.optimizer.Adam(ts.default_graph, loss, learning_rate)

batch_size = 8

for epoch in range(200):
    batch_count = 0
    for i in range(len(X)):
        x1.set_value(np.mat(X[i]).T)
        label.set_value(np.mat(y[i]))
        optimizer.one_step()
        batch_count += 1
        if batch_count >= batch_size:
            optimizer.update()
            batch_count = 0

    pred = []
    for i in range(len(X)):
        x1.set_value(np.mat(X[i]).T)
        label.set_value(np.mat(y[i]))
        predict.forward()
        pred.append(predict.value[0, 0])

    pred = (np.array(pred) > 0.5).astype(np.int) * 2 - 1
    accuracy = (y == pred).astype(np.int).sum() / len(X)
    print("epoch: {:d}, accuracy: {:.3f}".format(epoch + 1, accuracy))