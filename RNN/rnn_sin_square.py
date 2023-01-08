import sys
sys.path.append('../tensorslow')

import numpy as np
import tensorslow as ts
from scipy import signal

# Generate two sequential dataset
def get_sequence_data(dimension=10, length=10,
                      number_of_examples=1000, train_set_ratio=0.7, seed=42):
    xx = []

    # sinusoidal
    xx.append(np.sin(np.arange(0, 10, 10 / length)).reshape(-1, 1))

    # square
    xx.append(np.array(signal.square(np.arange(0, 10, 10 / length))).reshape(-1, 1))

    data = []
    for i in range(2):
        x = xx[i]
        for j in range(number_of_examples // 2):
            sequence = x + np.random.normal(0, 0.6, (len(x), dimension))  # 加入噪声
            label = np.array([int(i == k) for k in range(2)])
            data.append(np.c_[sequence.reshape(1, -1), label.reshape(1, -1)])

    data = np.concatenate(data, axis=0)

    np.random.shuffle(data)

    train_set_size = int(number_of_examples * train_set_ratio)  # 训练集样本数量

    return (data[:train_set_size, :-2].reshape(-1, length, dimension),
            data[:train_set_size, -2:],
            data[train_set_size:, :-2].reshape(-1, length, dimension),
            data[train_set_size:, -2:])

# construct RNN
# h_t = ReLU(Ux_t + Wh_t-1 + b)
seq_len = 96  
dimension = 16  # input dimension
status_dimension = 12 

signal_train, label_train, signal_test, label_test = get_sequence_data(length=seq_len, dimension=dimension)

inputs = [ts.core.Variable(dim=(dimension, 1), init=False, trainable=False) for i in range(seq_len)]

U = ts.core.Variable(dim=(status_dimension, dimension), init=True, trainable=True)

W = ts.core.Variable(dim=(status_dimension, status_dimension), init=True, trainable=True)

b = ts.core.Variable(dim=(status_dimension, 1), init=True, trainable=True)

last_step = None #h_t-1

for iv in inputs:
    h = ts.ops.Add(ts.ops.MatMul(U, iv), b)
    if last_step is not None:
        h = ts.ops.Add(ts.ops.MatMul(W, last_step), h)
    h = ts.ops.ReLU(h)
    last_step = h

fc1 = ts.layer.fc(last_step, status_dimension, 40, "ReLU")  
fc2 = ts.layer.fc(fc1, 40, 10, "ReLU")  
output = ts.layer.fc(fc2, 10, 2, "None")  

predict = ts.ops.SoftMax(output)
label = ts.core.Variable((2, 1), trainable=False)
loss = ts.ops.CrossEntropyWithSoftMax(output, label)

# train
learning_rate = 0.005
optimizer = ts.optimizer.Adam(ts.default_graph, loss, learning_rate)

batch_size = 16

for epoch in range(50):
    batch_count = 0   
    for i, s in enumerate(signal_train):
        for j, x in enumerate(inputs):
            x.set_value(np.mat(s[j]).T)
        
        label.set_value(np.mat(label_train[i, :]).T)
        
        optimizer.one_step()
        
        batch_count += 1
        if batch_count >= batch_size:
            print("epoch: {:d}, iteration: {:d}, loss: {:.3f}".format(epoch + 1, i + 1, loss.value[0, 0]))
            optimizer.update()
            batch_count = 0
        

    pred = []
    for i, s in enumerate(signal_test):
        for j, x in enumerate(inputs):
            x.set_value(np.mat(s[j]).T)
        predict.forward()
        pred.append(predict.value.A.ravel())
            
    pred = np.array(pred).argmax(axis=1)
    true = label_test.argmax(axis=1)
    
    accuracy = (true == pred).astype(np.int).sum() / len(signal_test)
    print("epoch: {:d}, accuracy: {:.5f}".format(epoch + 1, accuracy))