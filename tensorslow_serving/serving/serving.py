from .proto import serving_pb2, serving_pb2_grpc
import numpy as np
import tensorslow as ts
import time
from concurrent.futures import ThreadPoolExecutor
import grpc


class TensorSlowServingService(serving_pb2_grpc.TensorSlowServingServicer):

    def __init__(self, root_dir, model_file_name, weights_file_name):

        self.root_dir = root_dir
        self.model_file_name = model_file_name
        self.weights_file_name = weights_file_name

        saver = ts.trainer.Saver(self.root_dir)

        # load the computation graph and the parameters

        _, service = saver.load(model_file_name=self.model_file_name,
                                weights_file_name=self.weights_file_name)
        
        assert service is not None

        inputs = service.get('inputs', None)
        assert inputs is not None

        outputs = service.get('outputs', None)
        assert outputs is not None    

        # find the input and output nodes based on the service signature
        self.input_node = ts.get_node_from_graph(inputs['name'])
        assert self.input_node is not None
        assert isinstance(self.input_node, ts.Variable)   

        self.input_dim = self.input_node.dim

        self.output_node = ts.get_node_from_graph(outputs['name'])
        assert self.output_node is not None

    def Predict(self, predict_req, context):
        # data in protobuf format deserialized to NumPy Matrix
        inference_req = TensorSlowServingService.deserialize(predict_req)

        # calculate the predicted output
        inference_resp = self._inference(inference_req)

        # predicted output serialized to protobuf format
        predict_resp = TensorSlowServingService.serialize(inference_resp)

        return predict_resp   

    def _inference(self, inference_req):
        """
        Calculate the predicted output using forward propagation
        """
        inference_resp_mat_list = []

        for mat in inference_req:
            self.input_node.set_value(mat.T)
            self.output_node.forward()

            inference_resp_mat_list.append(self.output_node.value)

        return inference_resp_mat_list

    def deserialized(predict_req):
        """
        protobuf format data => NumPy matrix
        """
        infer_req_mat_list = []
        for proto_mat in predict_req.data:
            dim = tuple(proto_mat.dim)
            mat = np.mat(proto_mat.value, dtype=np.float32)
            mat = np.reshape(mat, dim)
            infer_req_mat_list.append(mat)

        return infer_req_mat_list   

    def serialize(inference_resp):
        """
        NumPy Matrix => protobuf format data
        """
        resp = serving_pb2.PredictResp()
        for mat in inference_resp:
            proto_mat = resp.data.add()
            proto_mat.value.extend(np.array(mat).flatten())
            proto_mat.dim.extend(list(mat.shape))

        return resp
    
class TensorSlowServer(object):

    """
    sets up a gRPC server to serve a machine learning model for inference.
    """

    def __init__(self, host, root_dir, model_file_name, weights_file_name, max_workers=10):

        self.host = host
        self.max_workers = max_workers

        # create a new gRPC server
        self.server = grpc.server(
            ThreadPoolExecutor(max_workers=self.max_workers))

        serving_pb2_grpc.add_TensorSlowServingServicer_to_server(
            TensorSlowServingService(root_dir, model_file_name, weights_file_name), self.server)

        # specify the listening address for the server
        self.server.add_insecure_port(self.host)  

    def serve(self):
        #start rpc server
        self.server.start()
        print('TensorSlow server running on {}'.format(self.host))       

        try:
            while True:
                time.sleep(60 * 60 * 24)  
        except KeyboardInterrupt:
            self.server.stop(0)         

            