import json
import os
import datetime

import numpy as np

from ..core.core import get_node_from_graph
from ..core import *
from ..core import Node, Variable
from ..core.graph import default_graph
from ..ops import *
from ..ops.loss import *
from ..ops.metrics import *
from ..util import ClassMining

class Saver(object):
    """
    Save two separate files
    1. Structure of the computation graph (json file)
    2. Paramters of the computation graph - Variable nodes (binary file)
    """

    def __init__(self, root_dir = ''):
        self.root_dir = root_dir
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)
    
    def save(self, graph=None, meta=None, service_signature=None,
             model_file_name='model.json',
             weights_file_name='weights.npz'):
        '''
        Save the computation graph into json and npz files
        '''
        if graph is None:
            graph = default_graph

        # meta information about the model: 
        meta = {} if meta is None else meta
        meta['save_time'] = str(datetime.datetime.now())
        meta['weights_file_name'] = weights_file_name

        service = {} if service_signature is None else service_signature

        # save
        self._save_model_and_weights(
            graph, meta, service, model_file_name, weights_file_name)
    
    def _save_model_and_weights(self, graph, meta, service, model_file_name, weights_file_name):
        model_json = {
            'meta': meta,
            'service': service
        }

        graph_json = []

        weights_dict = dict()

        for node in graph.nodes:
            if not node.need_save:
                continue
            node.kwargs.pop('name', None)
            node_json = {
                'node_type': node.__class__.__name__,
                'name': node.name,
                'parents': [parent.name for parent in node.parents],
                'children': [child.name for child in node.children],
                'kwargs': node.kwargs                
            }

            if node.value is not None:
                if isinstance(node.value, np.matrix):
                    node_json['dim'] = node.value.shape
            
            graph_json.append(node_json)

            if isinstance(node, Variable):
                # Also save the value
                weights_dict[node.name] = node.value
        
        model_json['graph'] = graph_json

        # save the model into model_file with json format
        model_file_path = os.path.join(self.root_dir, model_file_name)
        with open(model_file_path, 'w') as model_file:
            json.dump(model_json, model_file, indent=4)
            print('Save model into file: {}'.format(model_file_name))

        # save the variable nodes into weights_file with npz format
        weights_file_path = os.path.join(self.root_dir, weights_file_name)
        with open(weights_file_path, 'wb') as weights_file: # binary code
            np.savez(weights_file, **weights_dict)
            print('Save weights to file: {}'.format(weights_file.name))  

    def create_node(graph, from_model_json, node_json):
        '''
        create node recursively
        '''
        node_type = node_json['node_type']
        node_name = node_json['name']
        parents_name = node_json['parents']
        dim = node_json.get('dim', None)
        kwargs = node_json.get('kwargs', None)
        kwargs['graph'] = graph

        parents = []
        for parent_name in parents_name:
            parent_node = get_node_from_graph(parent_name, graph=graph)
            if parent_node is None:
                parent_node_json = None
                for node in from_model_json:
                    if node['name'] == parent_name:
                        parent_node_json = node

                assert parent_node_json is not None
                # create node recursively if there's no parent_node
                parent_node = Saver.create_node(
                    graph, from_model_json, parent_node_json)

            parents.append(parent_node)

        if node_type == 'Variable':
            assert dim is not None

            dim = tuple(dim)
            return ClassMining.get_instance_by_subclass_name(Node, node_type)(*parents, dim=dim, name=node_name, **kwargs)
        else:
            return ClassMining.get_instance_by_subclass_name(Node, node_type)(*parents, name=node_name, **kwargs)


    def _restore_nodes(self, graph, from_model_json, from_weights_dict):

        for index in range(len(from_model_json)):
            node_json = from_model_json[index]
            node_name = node_json['name']

            weights = None
            if node_name in from_weights_dict:
                weights = from_weights_dict[node_name]

            # update or create
            target_node = get_node_from_graph(node_name, graph=graph)
            if target_node is None:
                print('Target node {} of type {} not exists, try to create the instance'.format(
                    node_json['name'], node_json['node_type']))
                target_node = Saver.create_node(
                    graph, from_model_json, node_json)

            target_node.value = weights


    def load(self, to_graph=None,
             model_file_name='model.json',
             weights_file_name='weights.npz'):
        '''
        Restore the structure of the computation graph and values for the variable nodes
        '''
        if to_graph is None:
            to_graph = default_graph

        model_json = {}
        graph_json = []
        weights_dict = dict()

        # extract the model
        model_file_path = os.path.join(self.root_dir, model_file_name)
        with open(model_file_path, 'r') as model_file:
            model_json = json.load(model_file)

        # extract values for the weights
        weights_file_path = os.path.join(self.root_dir, weights_file_name)
        with open(weights_file_path, 'rb') as weights_file:
            weights_npz_files = np.load(weights_file)
            for file_name in weights_npz_files.files:
                weights_dict[file_name] = weights_npz_files[file_name]
            weights_npz_files.close()

        graph_json = model_json['graph']
        self._restore_nodes(to_graph, graph_json, weights_dict)
        print('Load and restore model from {} and {}'.format(
            model_file_path, weights_file_path))

        self.meta = model_json.get('meta', None)
        self.service = model_json.get('service', None)
        return self.meta, self.service        