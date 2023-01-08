import sys
sys.path.append('../tensorslow')

import numpy as np
import tensorslow as ts
from sklearn.datasets import make_classification

dimension = 60

X, y = make_classification(600, dimension, n_informative=20)
y = y * 2 - 1

# dimension of the embedding
k = 20

# x1 : 60 x 1
x1 = ts.core.Variable(dim=(dimension, 1), init=False, trainable=False)
label = ts.core.Variable(dim=(1, 1), init=False, trainable=False)
w = ts.core.Variable(dim=(1, dimension), init=True, trainable=True)
# E : 20 x 60
E = ts.core.Variable(dim=(k, dimension), init=True, trainable=True)
b = ts.core.Variable(dim=(1, 1), init=True, trainable=True)

# "wide" part
wide = ts.ops.MatMul(w, x1)

# "deep" part
embedding = ts.ops.MatMul(E, x1)

hidden_1 = ts.layer.fc(embedding, k, 8, "ReLU")
hidden_2 = ts.layer.fc(hidden_1, 8, 4, "ReLU")
deep = ts.layer.fc(hidden_2, 4, 1, None)

output = ts.ops.Add(wide, deep, b)
predict = ts.ops.Logistic(output)
loss = ts.ops.loss.LogLoss(ts.ops.Multiply(label, output))

learning_rate = 0.005
optimizer = ts.optimizer.Adam(ts.default_graph, loss, learning_rate)

batch_size = 16
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
        predict.forward()
        pred.append(predict.value[0, 0])  
    pred = (np.array(pred) > 0.5).astype(np.int) * 2 - 1
    accuracy = (y == pred).astype(np.int).sum() / len(X)

    print("epoch: {:d}, accuracy: {:.3f}".format(epoch + 1, accuracy))