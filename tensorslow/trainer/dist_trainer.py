from ..core import (Variable, get_trainable_variables_from_graph,
                    update_node_value_in_graph)
from ..core.graph import default_graph
from ..dist import ps
from .trainer import Trainer

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