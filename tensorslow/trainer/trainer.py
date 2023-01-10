import numpy as np
from ..core import Variable, default_graph

class Trainer(object):

    def __init__(self, input_x, input_y,
                 loss_op, optimizer,
                 epoches, batch_size=8,
                 eval_on_train=False, metrics_ops=None, *args, **kwargs):
        # list of inputs
        self.inputs = input_x

        # label
        self.input_y = input_y

        # loss function
        self.loss_op = loss_op

        self.optimizer = optimizer

        self.epoches = epoches
        self.epoch = 0

        self.batch_size = batch_size
        self.eval_on_train = eval_on_train

        self.metrics_ops = metrics_ops

        self.print_iteration_interval = kwargs.get('print_iteration_interval', 100)

    def train_and_eval(self, train_x, train_y, test_x=None, test_y=None):
        # Note: train_x is a dict. key: name of the input node; value: sample data
        # We can allow multiple input nodes
        assert len(train_x) == len(self.inputs)

        if test_x is not None and test_y is not None:
            assert len(test_x) == len(self.inputs)  
    
        # initialize the weights
        self._variable_weights_init()
        print('[INIT] Variable weights init finished')

        # main loop starts
        self.main_loop(train_x, train_y, test_x, test_y)

    def main_loop(self, train_x, train_y, test_x, test_y):
        for self.epoch in range(self.epoches):
            self.train(train_x, train_y)
            if self.eval_on_train and test_x is not None and test_y is not None:
                self.eval(test_x, test_y)
    
    def train(self, train_x, train_y):
        """
        for each sample, apply forward and backward propagation to obtain the loss and the jacobian
        udpate the optimizer when the training set size reaches the batch size
        """
        for i in range(len(list(train_x.values())[0])):
            self.one_step(self._get_input_values(train_x, i), train_y[i])
            if (i + 1) % self.batch_size == 0:
                self._optimizer_update

    def _variable_weights_init(self):
        raise NotImplementedError()

    def _optimizer_update(self):
        raise NotImplementedError()
            
    def _get_input_values(self, x, index):
        # get the {index}th sample from x
        input_values = dict()
        for input_node_name in x.keys():
            input_values[input_node_name] = x[input_node_name][index]

        return input_values
    
    def one_step(self, data_x, data_y, is_eval=False):
        for i in range(len(self.inputs)):
            input_value = data_x.get(self.inputs[i].name)
            self.inputs[i].set_value(np.mat(input_value).T)

        self.input_y.set_value(np.mat(data_y).T)

        if not is_eval:
            self.optimizer.one_step()
     
    def eval(self, test_x, test_y):
        for metrics_op in self.metrics_ops:
            metrics_op.reset()

        for i in range(len(list(test_x.values())[0])):
            self.one_step(self._get_input_values(
                test_x, i), test_y[i], is_eval=True)
            for metrics_op in self.metrics_ops:
                metrics_op.forward()

        metrics_str = 'Epoch [{}] evaluation metrics '.format(self.epoch + 1)
        for metrics_op in self.metrics_ops:
            metrics_str += metrics_op.value_str()

        print(metrics_str)