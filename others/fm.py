import sys
sys.path.append('../tensorslow')

import numpy as np
import tensorslow as ts
from sklearn.datasets import make_circles

X, y = make_circles(600, noise=0.1, factor=0.2)
y = y * 2 - 1

# dimension of the feature
dimension = 20

# noise
X = np.concatenate([X, np.random.normal(0.0, 0.5, (600, dimension-2))], axis=1)

# dimension of the hidden vector
k = 2

x1 = ts.core.Variable(dim=(dimension, 1), init=False, trainable=False)
label = ts.core.Variable(dim=(1, 1), init=False, trainable=False)
w = ts.core.Variable(dim=(1, dimension), init=True, trainable=True)

# W for quadratic features
H = ts.core.Variable(dim=(k, dimension), init=True, trainable=True)
HTH = ts.ops.MatMul(ts.ops.Reshape(H, shape=(dimension, k)), H)

b = ts.core.Variable(dim=(1, 1), init=True, trainable=True)

output = ts.ops.Add(
        ts.ops.MatMul(w, x1),   
        ts.ops.MatMul(ts.ops.Reshape(x1, shape=(1, dimension)),
                      ts.ops.MatMul(HTH, x1)),
        b)

predict = ts.ops.Logistic(output)

loss = ts.ops.loss.LogLoss(ts.ops.Multiply(label, output))

learning_rate = 0.001
optimizer = ts.optimizer.Adam(ts.default_graph, loss, learning_rate)

batch_size = 16

for epoch in range(50):
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
        predict.forward()
        pred.append(predict.value[0, 0])
            
    pred = (np.array(pred) > 0.5).astype(np.int) * 2 - 1
    accuracy = (y == pred).astype(np.int).sum() / len(X)
       
    print("epoch: {:d}, accuracy: {:.3f}".format(epoch + 1, accuracy))