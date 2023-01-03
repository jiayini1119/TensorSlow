
# optimize the adaline with Mini Batch Gradient Descent

import numpy as np
import tensorslow as ts

male_heights = np.random.normal(171, 6, 500)
female_heights = np.random.normal(158, 5, 500)

male_weights = np.random.normal(70, 10, 500)
female_weights = np.random.normal(57, 8, 500)

male_bfrs = np.random.normal(16, 2, 500)
female_bfrs = np.random.normal(22, 2, 500)

# labels
male_labels = [1] * 500
female_labels = [-1] * 500

# training set: 1000 x 4 numpy arrays:
# height   weight    bfrs   label
train_set = np.array([np.concatenate((male_heights, female_heights)),
                      np.concatenate((male_weights, female_weights)),
                      np.concatenate((male_bfrs, female_bfrs)),
                      np.concatenate((male_labels, female_labels))]).T

np.random.shuffle(train_set)

batch_size = 10

# build computation graph

X = ts.core.Variable(dim=(batch_size, 3), init=False, trainable=False)
label = ts.core.Variable(dim=(batch_size, 1), init=False, trainable=False)

# We want to train w and b
w = ts.core.Variable(dim=(3, 1), init=True, trainable=True)
b = ts.core.Variable(dim=(1, 1), init=True, trainable=True)

ones = ts.core.Variable(dim=(batch_size, 1), init=False, trainable=False)
ones.set_value(np.mat(np.ones(batch_size)).T)
bias = ts.ops.ScalarMultiply(b, ones)  # (batch_size) duplicates of b

# prediction
output = ts.ops.Add(ts.ops.MatMul(X, w), bias)
predict = ts.ops.Step(output)

loss = ts.ops.loss.PerceptionLoss(ts.ops.Multiply(label, output))

# Average loass for one mini batch of training set
B = ts.core.Variable(dim=(1, batch_size), init=False, trainable=False)
B.set_value(1 / batch_size * np.mat(np.ones(batch_size)))
mean_loss = ts.ops.MatMul(B, loss)

learning_rate = 0.0001

# training
for epoch in range(50):

    for i in np.arange(0, len(train_set), batch_size):
        # feathers for a mini batch
        features = np.mat(train_set[i:i + batch_size, :-1])
        # labels for a mini batch
        l = np.mat(train_set[i:i + batch_size, -1]).T
        X.set_value(features)
        label.set_value(l)

        # forward propagation on the mean_loss node
        mean_loss.forward()

        # back propagation on the parameters
        w.backward(mean_loss)
        b.backward(mean_loss)

        # udpate
        w.set_value(w.value - learning_rate * w.jacobi.T.reshape(w.shape()))
        b.set_value(b.value - learning_rate * b.jacobi.T.reshape(b.shape()))

        ts.default_graph.clear_jacobi()

    # feedback
    pred = []

    for i in np.arange(0, len(train_set), batch_size):

        features = np.mat(train_set[i:i + batch_size, :-1])
        X.set_value(features)

       # forward propagation on the predict node
        predict.forward()

        # prediction for a mini-batch size of the sample! Not one sample.
        pred.extend(predict.value.A.ravel())

    pred = np.array(pred) * 2 - 1
    accuracy = (train_set[:, -1] == pred).astype(np.int).sum() / len(train_set)
    print("epoch: {:d}, accuracy: {:.3f}".format(epoch + 1, accuracy))
