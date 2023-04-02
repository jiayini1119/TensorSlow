import sys
sys.path.append('../tensorslow')

import numpy as np
import argparse
from tensorslow.dist.ps import ps
# from tensorslow_serving.exporter import Exporter
from tensorslow.trainer import DistTrainerParameterServer
import tensorslow as ts
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import fetch_openml

cluster_conf = {
    "ps": [
        "localhost:5000"
    ],
    "workers": [
        "localhost:6000",
        "localhost:6002",
        "localhost:6004"
    ]
}

def train(worker_index):
    img_shape = (28, 28)
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, cache=True)

    start = worker_index * 1000
    end = (worker_index + 1)* 1000

    X, y = X[start:end] / 255, y.astype(np.int)[start:end]
    X = np.reshape(np.array(X), (1000, *img_shape))

    oh = OneHotEncoder(sparse=False)
    one_hot_label = oh.fit_transform(y.reshape(-1, 1))

    x = ts.core.Variable(img_shape, init=False,
                         trainable=False, name='img_input')

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

    # use DistTrainerParameterServer as the trainer
    trainer = DistTrainerParameterServer([x], one_hot, loss, optimizer,
                                         epoches=10, batch_size=32,
                                         eval_on_train=True, metrics_ops=[accuracy],
                                         cluster_conf=cluster_conf, worker_index=worker_index)
    
    trainer.train_and_eval({x.name: X}, one_hot_label,
                           {x.name: X}, one_hot_label)
    
if __name__ == 'main':
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', type=str)
    parser.add_argument('--worker_index', type=int)

    args = parser.parse_args()

    role = args.role


    if role == 'ps':
        server = ps.ParameterServiceServer(cluster_conf, sync=True)
        server.serve()

    else: # worker
        worker_index = args.worker_index
        train(worker_index)