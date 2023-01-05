import sys
sys.path.append('../tensorslow')

# Step function => logistic function
# PerceptionLoss => LogLoss

import tensorslow as ts
import numpy as np

male_heights = np.random.normal(171, 6, 500)
female_heights = np.random.normal(158, 5, 500)

male_weights = np.random.normal(70, 10, 500)
female_weights = np.random.normal(57, 8, 500)

male_bfrs = np.random.normal(16, 2, 500)
female_bfrs = np.random.normal(22, 2, 500)

male_labels = [1] * 500
female_labels = [-1] * 500

train_set = np.array([np.concatenate((male_heights, female_heights)),
                      np.concatenate((male_weights, female_weights)),
                      np.concatenate((male_bfrs, female_bfrs)),
                      np.concatenate((male_labels, female_labels))]).T

np.random.shuffle(train_set)

x = ts.core.Variable(dim=(3, 1), init=False, trainable=False)

label = ts.core.Variable(dim=(1, 1), init=False, trainable=False)

w = ts.core.Variable(dim=(1, 3), init=True, trainable=True)

b = ts.core.Variable(dim=(1, 1), init=True, trainable=True)

output = ts.ops.Add(ts.ops.MatMul(w, x), b)
predict = ts.ops.Logistic(output)
loss = ts.ops.loss.LogLoss(ts.ops.Multiply(label, output))

learning_rate = 0.0001

optimizer = ts.optimizer.Adam(ts.default_graph, loss, learning_rate)

batch_size = 16

for epoch in range(50):
    batch_count = 0
    for i in range(len(train_set)):
        features = np.mat(train_set[i, :-1]).T
        l = np.mat(train_set[i, -1])
        x.set_value(features)
        label.set_value(l)
        # one step of forward + backward propagation
        optimizer.one_step()
        batch_count += 1
        if batch_count >= batch_size:
            optimizer.update()
            batch_count = 0

    pred = []
    for i in range(len(train_set)):
        features = np.mat(train_set[i, :-1]).T
        x.set_value(features)
        predict.forward()
        pred.append(predict.value[0, 0])  

    pred = (np.array(pred) > 0.5).astype(np.int) * 2 - 1
    accuracy = (train_set[:, -1] == pred).astype(np.int).sum() / len(train_set)
    print("epoch: {:d}, accuracy: {:.3f}".format(epoch + 1, accuracy))
