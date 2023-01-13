import tensorslow as ts

class Exporter(object):
    """
    export the model
    """

    def __init__(self, graph=None):
        self.graph = ts.default_graph if graph is None else graph
    
    def signature(self, input_name, output_name):
        input_var = ts.get_node_from_graph(input_name, graph=self.graph)
        assert input_var is not None
        output_var = ts.get_node_from_graph(output_name, graph=self.graph)
        assert output_var is not None

        input_signature = dict()
        input_signature['name'] = input_var.name

        output_signature = dict()
        output_signature['name'] = output_var.name
        return {
            'inputs': input_signature,
            'outputs': output_signature
        }   
 
    
