# model saving

import sys
sys.path.append('../tensorslow')

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
import tensorslow as ts
from tensorslow.trainer import SimpleTrainer
from tensorslow_serving.exporter import Exporter

# build computation graph
img_shape = (28, 28)

X, y = fetch_openml('mnist_784', version=1, return_X_y=True, cache=True)
X, y = X[:1000] / 255, y.astype(np.int)[:1000]
X = np.reshape(np.array(X), (1000, *img_shape))

oh = OneHotEncoder(sparse=False)
one_hot_label = oh.fit_transform(y.reshape(-1, 1))

x = ts.core.Variable(img_shape, init=False, trainable=False, name='img_input')

one_hot = ts.core.Variable(dim=(10, 1), init=False, trainable=False)

conv1 = ts.layer.conv([x], img_shape, 3, (5, 5), "ReLU")

pooling1 = ts.layer.pooling(conv1, (3, 3), (2, 2))

conv2 = ts.layer.conv(pooling1, (14, 14), 3, (3, 3), "ReLU")

pooling2 = ts.layer.pooling(conv2, (3, 3), (2, 2))

fc1 = ts.layer.fc(ts.ops.Concat(*pooling2), 147, 120, "ReLU")

output = ts.layer.fc(fc1, 120, 10, "None")

predict = ts.ops.SoftMax(output, name='softmax_output')

loss = ts.ops.loss.CrossEntropyWithSoftMax(output, one_hot)

learning_rate = 0.05

optimizer = ts.optimizer.RMSProp(ts.default_graph, loss, learning_rate)

accuracy = ts.ops.metrics.Accuracy(output, one_hot)

# training
trainer = SimpleTrainer(
    [x], one_hot, loss, optimizer, epoches=10, batch_size=32,
    eval_on_train=True, metrics_ops=[accuracy])

trainer.train_and_eval({x.name: X}, one_hot_label, {x.name: X}, one_hot_label)

# define server signature
exporter = Exporter()
sig = exporter.signature('img_input', 'softmax_output')

saver = ts.trainer.Saver('./epoches')

saver.save(model_file_name='my_model.json',
           weights_file_name='my_weights.npz', service_signature=sig)
