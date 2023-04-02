# load model and make prediction

import sys
sys.path.append('../tensorslow')
import numpy as np
from sklearn.datasets import fetch_openml
import tensorslow as ts

img_shape = (28, 28)

test_data, test_label = fetch_openml(
    'mnist_784', version=1, return_X_y=True, cache=True)
test_data, test_label = test_data[1000:2000] / \
    255, test_label.astype(np.int)[1000:2000]
test_data = np.reshape(np.array(test_data), (1000, *img_shape))


saver = ts.trainer.Saver('./epoches')

saver.load(model_file_name='my_model.json', weights_file_name='my_weights.npz')

x = ts.get_node_from_graph('img_input')
pred = ts.get_node_from_graph('softmax_output')

for index in range(len(test_data)):
    x.set_value(np.mat(test_data[index]).T)
    pred.forward()
    gt = test_label[index]
    print('model predict {} and ground truth: {}'.format(
        np.argmax(pred.value), gt))
