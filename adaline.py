
# build computation graph for Adaline and train the model

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

# build computation graph

x = ts.core.Variable(dim=(3, 1), init=False, trainable=False)
label = ts.core.Variable(dim=(1, 1), init=False, trainable=False)

# We want to train w and b
w = ts.core.Variable(dim=(1, 3), init=True, trainable=True)
b = ts.core.Variable(dim=(1, 1), init=True, trainable=True)

# prediction
output = ts.ops.Add(ts.ops.MatMul(w, x), b)
predict = ts.ops.Step(output)

loss = ts.ops.loss.PerceptionLoss(ts.ops.MatMul(label, output))

# train
learning_rate = 0.0001

for epoch in range(50):
    for i in range(len(train_set)):
        features = np.mat(train_set[i, :-1]).T
        l = np.mat(train_set[i, -1])

        x.set_value(features)
        label.set_value(l)

        loss.forward()

        w.backward(loss)
        b.backward(loss)

        # gradient descent
        new_w = w.value - learning_rate * w.jacobi.T.reshape(w.shape())
        new_b = b.value - learning_rate * b.jacobi.T.reshape(b.shape())
        w.set_value(new_w)
        b.set_value(new_b)

        ts.default_graph.clear_jacobi()

    # feedback
    pred = []
    for i in range(len(train_set)):
        features = np.mat(train_set[i, :-1]).T
        x.set_value(features)
        predict.forward()
        pred.append(predict.value[0, 0])

    pred = np.array(pred) * 2 - 1  # change from 1/0 => 1/-1
    hit = (train_set[:, -1] == pred).astype(np.int).sum()
    accuracy = hit / len(train_set)

    print("epoch: {:d}, accuracy: {:.3f}".format(epoch + 1, accuracy))
