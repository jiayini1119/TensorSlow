import sys
sys.path.append('../tensorslow')

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import tensorslow as ts

data = pd.read_csv("data/Iris.csv").drop("Id", axis=1)

data = data.sample(len(data), replace=False)

le = LabelEncoder()
number_label = le.fit_transform(data["Species"])

# One-Hot encoding of labels
oh = OneHotEncoder(sparse=False)
one_hot_label = oh.fit_transform(number_label.reshape(-1, 1))

features = data[['SepalLengthCm',
                 'SepalWidthCm',
                 'PetalLengthCm',
                 'PetalWidthCm']].values

# input: 4 x 1
x = ts.core.Variable(dim=(4, 1), init=False, trainable=False)
# One Hot: 3 x 1
one_hot = ts.core.Variable(dim=(3, 1), init=False, trainable=False)

################################################
hidden_1 = ts.layer.fc(x, 4, 10, "ReLU")

hidden_2 = ts.layer.fc(hidden_1, 10, 10, "ReLU")

# output layer
output = ts.layer.fc(hidden_2, 10, 3, None)

# # weight: 3 x 4
# W = ts.core.Variable(dim=(3, 4), init=True, trainable=True)
# # bias: 3 x 1
# b = ts.core.Variable(dim=(3, 1), init=True, trainable=True)

# linear = ts.ops.Add(ts.ops.MatMul(W, x), b) # logit

predict = ts.ops.SoftMax(output)

# cross entropy loss
loss = ts.ops.loss.CrossEntropyWithSoftMax(output, one_hot)

learning_rate = 0.02

# Construct optimizer
optimizer = ts.optimizer.Adam(ts.default_graph, loss, learning_rate)
batch_size = 16

for epoch in range(30):
    batch_count = 0
    for i in range(len(features)):
        feature = np.mat(features[i,:]).T
        label = np.mat(one_hot_label[i,:]).T
        x.set_value(feature)
        one_hot.set_value(label)
        optimizer.one_step()
        batch_count += 1
        if batch_count >= batch_size:
            optimizer.update()
            batch_count = 0
            

    pred = []
    for i in range(len(features)):
        feature = np.mat(features[i,:]).T
        x.set_value(feature)
        predict.forward()
        pred.append(predict.value.A.ravel())  
    # get the type with the highest probability as the predicted type
    pred = np.array(pred).argmax(axis=1)
    accuracy = (number_label == pred).astype(np.int).sum() / len(data)
    print("epoch: {:d}, accuracy: {:.3f}".format(epoch + 1, accuracy))