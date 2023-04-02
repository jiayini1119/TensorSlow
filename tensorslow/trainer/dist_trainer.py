from ..core import (Variable, get_trainable_variables_from_graph,
                    update_node_value_in_graph)
from ..core.graph import default_graph
from ..dist import ps, allreduce
from .trainer import Trainer
import threading

class DistTrainerParameterServer(Trainer):

    def __init__(self, *args, **kargs):
        Trainer.__init__(self, *args, **kargs)
        cluster_conf = kargs['cluster_conf']
        ps_host = cluster_conf['ps'][0]
        self.ps_client = ps.ParameterServiceClient(ps_host)

    def _variable_weights_init(self): # consistent initialization
        var_weights_dict = dict()
        for node in default_graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                var_weights_dict[node.name] = node.value
        
        duplicated_var_weights_dict = self.ps_client.variable_weights_init(var_weights_dict)

        for var_name, weights in duplicated_var_weights_dict.items():
            update_node_value_in_graph(var_name, weights)
        

    def _optimizer_update(self):
        acc_gradient = self.optimizer.acc_gradient
        # push
        self.ps_client.push_gradients(acc_gradient, self.optimizer.acc_no)
        # pull
        node_gradients_dict = self.ps_client.pull_gradients()

        self.optimizer.update(node_gradients_dict)

class DistTrainerRingAllReduce(Trainer):
    def __init__(self, *args, **kargs):
        Trainer.__init__(self, *args, **kargs)

        self.cluster_conf = kargs['cluster_conf']
        self.worker_index = kargs['worker_index']

        self.workers = self.cluster_conf['workers']
        self.worker_num = len(self.workers)
        self.host = self.workers[self.worker_index]

        self.step = self.worker_num - 1

        # right neighbour
        self.target_host = self.workers[(
            self.worker_index + 1) % self.worker_num]

        self.is_init = False
        self.init_cond = threading.Condition()

        self.cur_partition_index = self.worker_index
        self.partition = []

        # All trainable nodes
        self.variables = get_trainable_variables_from_graph()

        # patition the variables
        self._partition_variables()

        # for sending and receiving the gradients
        self.is_recieved = False
        self.recieved_gradients = None
        self.recieved_acc_no = None
        self.cond = threading.Condition()

        # server
        allreduce.RingAllReduceServer(
            self.host, self.worker_index,
            self._variable_weights_init_callback,
            self._scatter_callback,
            self._gather_callback).serve()

        # client
        self.client = allreduce.RingAllReduceClient(self.target_host)

    def _variable_weights_init(self):

        var_weights_dict = dict()
        for node in default_graph.nodes:
            if isinstance(node, Variable) and node.trainable:
                var_weights_dict[node.name] = node.value
        print('[INIT] Send variable init weights to worker ', self.target_host)

        if self.worker_index == 0: # first worker just uses the default value
            self.client.variable_weights_init(var_weights_dict)
        else: # wait for its left neighbour to be initialized first
            self.init_cond.acquire()
            while not self.is_init:
                self.init_cond.wait()
            self.init_cond.release()
            self.client.variable_weights_init(var_weights_dict) 

    def _variable_weights_init_callback(self, var_weights_dict):

        if self.worker_index != 0:
            print('[INIT] Variables initializing weights from last worker node...')
            for var_name, weights in var_weights_dict.items():
                update_node_value_in_graph(var_name, weights)

        # finish initialization
        self.init_cond.acquire()
        self.is_init = True
        self.init_cond.notify_all()
        self.init_cond.release()  

    
    def _optimizer_update(self):

        # scatter (N - 1) times
        for scatter_index in range(self.step):
            gradients_part = self._get_gradients_partition()
            cur_acc_no = self.optimizer.acc_no if scatter_index == 0 else self.recieved_acc_no

            # send the scattered block to its right neighbour
            self.client.send(gradients_part, cur_acc_no, 'scatter')

            # wait for the scattered block from the left neighbour
            self._wait_for_recieve('scatter')

        # All-gather (N - 1) times
        for gather_index in range(self.step):
            gradients_part = self._get_gradients_partition()
            self.client.send(gradients_part, 0, 'gather')
            self._wait_for_recieve('gather')

        self.optimizer.update()

    def _partition_variables(self):
        return


    def _get_gradients_partition(self):
        return


    def _scatter_callback(self):
        return

    def _gather_callback(self):
        return 

    def _wait_for_recieve(self):
        return