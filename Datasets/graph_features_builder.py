import json
import time
from random import random

import networkx as nx
import pandas as pd

from config import PROGRAM_GRAPH_FUNCTION_NODE_MANGLED_NAME, PROGRAM_GRAPH_FUNCTION_NODE_DEMANGLED_NAME
import numpy as np

from experiments.data_exploration.exploration_config import PROGRAM_ID


def get_nodes_attribute_from_networkx_graph(nodes_data_list, node_label_name: str):
    label_function_names_by_graph_node_order = []
    for node_info in nodes_data_list:
        node_name = node_info[0]
        node_data = node_info[1]
        mangled_name = node_data[node_label_name]
        label_function_names_by_graph_node_order.append((node_name, mangled_name))
    return label_function_names_by_graph_node_order


def get_mangled_and_demangled_names_from_networkx_graph(graph):
    nodes_list = graph.nodes.data()
    mangled_function_names_by_graph_node_order = get_nodes_attribute_from_networkx_graph(nodes_list,
                                                                                         PROGRAM_GRAPH_FUNCTION_NODE_MANGLED_NAME)
    demangled_function_names_by_graph_node_order = get_nodes_attribute_from_networkx_graph(nodes_list,
                                                                                           PROGRAM_GRAPH_FUNCTION_NODE_DEMANGLED_NAME)
    return mangled_function_names_by_graph_node_order, demangled_function_names_by_graph_node_order


def demangled_name(graph: nx.Graph, node_id: str):
    return graph.nodes[node_id][PROGRAM_GRAPH_FUNCTION_NODE_DEMANGLED_NAME]


def mangled_name(graph: nx.Graph, node_id: str):
    return graph.nodes[node_id][PROGRAM_GRAPH_FUNCTION_NODE_MANGLED_NAME]


def out_degree(graph: nx.DiGraph, node_id: str):
    return graph.out_degree[node_id]


def in_degree(graph: nx.DiGraph, node_id: str):
    return graph.in_degree[node_id]


def avg_num_of_in_params(graph: nx.DiGraph, node_id: str):
    number_valid_succs = 0
    sum_generic_successor = 0
    for i, j in graph.edges:
        if int(j) == int(node_id):
            try:
                num_generic_successors = len(
                    json.loads(graph.edges[(i, j)]['call_atts'].replace('\'', '"'))['generic_successors'])
                sum_generic_successor += num_generic_successors
                number_valid_succs += 1
            except:
                number_valid_succs += 0
    if number_valid_succs != 0:
        return sum_generic_successor / number_valid_succs
    return -1


def avg_num_of_out_params(graph: nx.DiGraph, node_id: str):
    number_valid_succs = 0
    sum_generic_successor = 0
    for i, j in graph.edges:
        if int(i) == int(node_id):
            try:
                num_generic_successors = len(
                    json.loads(graph.edges[(i, j)]['call_atts'].replace('\'', '"'))['generic_successors'])
                sum_generic_successor += num_generic_successors
                number_valid_succs += 1
            except:
                number_valid_succs += 0
    if number_valid_succs != 0:
        return sum_generic_successor / number_valid_succs
    return -1

def random_feature(graph: nx.DiGraph, node_id: str):
    time.sleep(0.00001)
    return random()

class GraphFeaturesBuilder:
    features_dict = {
        'demangled_name': demangled_name,
        'mangled_name': mangled_name,
        'out_degree': out_degree,
        'in_degree': in_degree,
        'num_in_params': avg_num_of_in_params,
        'num_out_params': avg_num_of_out_params,
        'random_feature': random_feature
    }

    def __init__(self, graph: nx.Graph, features_names_list, program_name):
        self.features_names_list = features_names_list
        self._graph = graph
        self.features_funcs_list = []
        self._program_name = program_name
        for name in self.features_names_list:
            self.features_funcs_list.append(self.get_feature_func_from_feature_name(name))

    def get_nodes_by_order(self):
        '''
        as long as no nodes were added, the graph nodes in the *same* certain order
        :return:
        '''
        return sorted(self._graph.nodes())

    def get_node_labels(self):
        return []

    def generate_nodes_features(self) -> pd.DataFrame:
        nodes_ids = self.get_nodes_by_order()
        dict_of_features_values = {}
        dict_of_features_values['id'] = nodes_ids
        dict_of_features_values[PROGRAM_ID] = self._program_name
        for feature_func, feature_name in zip(self.features_funcs_list, self.features_names_list):
            dict_of_features_values[feature_name] = feature_func(nodes_ids)
        return pd.DataFrame(dict_of_features_values)

    def get_feature_func_from_feature_name(self, feature_name):
        '''
        each function in the dictionary should return a function that gets a graph
        and a node and the graph and return a feature value
        :param feature_name:
        :return:
        '''
        if feature_name in self.features_dict:
            def feature_name_of_node(nodes_id_list):
                return [self.features_dict[feature_name](self._graph, node_id) for node_id in nodes_id_list]

            return feature_name_of_node
        raise NotImplemented()
