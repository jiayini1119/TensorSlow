import threading
from ..dist import DistCommon
from ..proto import parameter_server_pb2 as pspb
from ..proto import parameter_server_pb2_grpc as psrpc

class ParameterService(psrpc.ParameterServiceServicer):
    def __init__(self, worker_num, sync=True):
        self.node_gradients_cache = dict() # { node_name : mean gradient }
        self.variable_weights_cache = dict()
        self.acc_no = 0 # batch size

        self.sync = sync
        self.worker_num = worker_num
        self.cur_push_num = 0
        self.cur_pull_num = self.worker_num

        # mutex
        self.cond = threading.Condition()
        self.push_lock = threading.Lock()
        self.init_lock = threading.Lock()
        self.is_init = False

    # Push
    def Push(self, push_req, context):
        node_with_gradients, acc_no = self._deserialize_push_req(push_req)

        if self.sync:
            self._push_sync(node_with_gradients, acc_no)
        else:
            self._push_async(node_with_gradients, acc_no)

        return pspb.ParameterPushResp()

    def _push_sync(self, node_with_gradients, acc_no):
        # lock
        if self.cond.acquire():
            # wait for all the workers to finish pulling
            while self.cur_pull_num != self.worker_num:
                self.cond.wait()

            self.cur_push_num += 1

            # update gradient
            self._update_gradients_cache(node_with_gradients)

            self.acc_no += acc_no

            # All workers finish pushing; notify all workers to pull the gradient
            if self.cur_push_num >= self.worker_num:
                self.cur_pull_num = 0
                self.cond.notify_all()
            self.cond.release() # unlock
        else:
            self.cond.wait()
    
    def _push_async(self, node_with_gradients, acc_no):
        self.push_lock.acquire()
        self._update_gradients_cache(node_with_gradients)
        self.acc_no += acc_no
        self.push_lock.release()

    # Pull
    def Pull(self, pull_req, context):
        if self.sync:
            resp = self._pull_sync()
        else:
            resp = self._pull_async()
        return resp
    

    def _pull_sync(self):
        # lock
        if self.cond.acquire():
            # wait for all the workers to finish pushing
            while self.cur_push_num != self.worker_num:
                self.cond.wait()

            self.cur_pull_num += 1
            
            # calculate gradient mean
            self._gradients_cache_mean()

            resp = self._serialize_pull_resp()

            # All workers finish pulling; notify all workers to push
            if self.cur_pull_num >= self.worker_num:
                self.cur_push_num = 0
                self._reset_gradients_cache()
                self.cond.notify_all()

            self.cond.release() # unlock
        else:
            self.cond.wait()

        return resp

    def _deserialize_push_req(self, push_req):
        acc_no = push_req.node_gradients.acc_no
        node_with_gradients = DistCommon._deserialize_proto_node_gradients(
            push_req.node_gradients)
        return node_with_gradients, acc_no
    
    def _serialize_pull_resp(self):
        proto_node_gradients = DistCommon._serialize_proto_node_gradients(
            self.node_gradients_cache)
        resp = pspb.ParameterPullResp(node_gradients=proto_node_gradients)
        return resp

    def _update_gradients_cache(self, node_with_gradients):
        for node_name, gradient in node_with_gradients.items():
            if node_name in self.node_gradients_cache:
                exists_gradient = self.node_gradients_cache[node_name]
                assert exists_gradient.shape == gradient.shape
                self.node_gradients_cache[node_name] = exists_gradient + gradient
            else:
                self.node_gradients_cache[node_name] = gradient
    
    def _gradients_cache_mean(self):
        if self.acc_no != 0:
            for node_name, gradient in self.node_gradients_cache.items():
                self.node_gradients_cache[node_name] = self.node_gradients_cache[node_name] / self.acc_no
            
            self.acc_no = 0 # reset
    
    def _reset_gradients_cache(self):
        self.node_gradients_cache.clear()

    def VariableWeightsInit(self, variable_weights_req, context):
        """
        Multiple workers simultaneously send their initial values to the ps;
        the ps uses the initial value of the first arriving worker and returns it to the other workers.
        """
        self.init_lock.acquire() # lock
        if not self.is_init:
            self.variable_weights_cache = DistCommon._deserialize_proto_variable_weights(
                variable_weights_req)
            print('Parameter service variable weights initialized')
        
        resp = DistCommon._serialize_proto_variable_weights(
            self.variable_weights_cache)
        self.is_init = True
        self.init_lock.release() # unlock

        return resp