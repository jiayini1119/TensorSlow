import sys
sys.path.append('../tensorslow/')

from tensorslow.trainer import SimpleTrainer
import tensorslow as ts
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import fetch_openml
import numpy as np

X, y = fetch_openml('mnist_784', version=1, return_X_y=True, cache=True)
X, y = X[:1000] / 255, y.astype(np.int)[:1000]
X = np.reshape(np.array(X), (1000, 28, 28))

oh = OneHotEncoder(sparse=False)
one_hot_label = oh.fit_transform(y.reshape(-1, 1))

img_shape = (28, 28)

x = ts.core.Variable(img_shape, init=False, trainable=False)

one_hot = ts.core.Variable(dim=(10, 1), init=False, trainable=False)

conv1 = ts.layer.conv([x], img_shape, 3, (5, 5), "ReLU")

pooling1 = ts.layer.pooling(conv1, (3, 3), (2, 2))

conv2 = ts.layer.conv(pooling1, (14, 14), 3, (3, 3), "ReLU")

pooling2 = ts.layer.pooling(conv2, (3, 3), (2, 2))

fc1 = ts.layer.fc(ts.ops.Concat(*pooling2), 147, 120, "ReLU")

output = ts.layer.fc(fc1, 120, 10, "None")

predict = ts.ops.SoftMax(output)

loss = ts.ops.loss.CrossEntropyWithSoftMax(output, one_hot)

learning_rate = 0.005

optimizer = ts.optimizer.Adam(ts.default_graph, loss, learning_rate)

batch_size = 32

trainer = SimpleTrainer(
    [x], one_hot, loss, optimizer, epoches=10, batch_size=batch_size)

trainer.train({x.name: X}, one_hot_label)