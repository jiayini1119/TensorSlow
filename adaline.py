
# build computation graph for Adaline and train the model

import numpy as np
import matrixslow as ms


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

x = ms.core.Variable(dim=(3, 1), init=False, trainable=False)
label = ms.core.Variable(dim=(1, 1), init=False, trainable=False)

# We want to train w and b
w = ms.core.Variable(dim=(1, 3), init=True, trainable=True)
b = ms.core.Variable(dim=(1, 1), init=True, trainable=True)

# prediction
output = ms.ops.Add(ms.ops.MatMul(w, x), b)
predict = ms.ops.Step(output)

loss = ms.ops.loss.PerceptionLoss(ms.ops.MatMul(label, output))
