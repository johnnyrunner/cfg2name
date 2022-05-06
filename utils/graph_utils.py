import itertools
import json
import logging
import os
import pickle
from pathlib import Path
from shutil import copyfile

from typing import Dict

import cpp_demangle
import networkx
import numpy
from jsonlines import jsonlines
from matplotlib import pyplot as plt
from networkx import Graph
import numpy as np
from tqdm import tqdm

from config import ORIGINAL_FUNCTIONS_FEATURES_FOLDER_NAME, \
    DIRE_PROGRAM_DATA_FILE_EXTENSION, FUNCTION_CALL_GRAPHS_DIRECTORY, PROGRAM_CALL_GRAPHS_FILE_EXTENSION, \
    PROGRAM_GRAPH_FUNCTION_NODE_DEMANGLED_NAME, DEMANGLED_NAME, GRAPH_ADDR_TO_FUNC_NAME_FILENAME_SUFFIX, \
    STRIPPED_FUNCTIONS_FEATURES_FOLDER_NAME, PROGRAM_GRAPH_FUNCTION_NODE_MANGLED_NAME, \
    FUNCTION_DOESNT_APPEAR_IN_BOTH_GRAPHS_DEMANGLED_NAME_PLACEHOLDER, \
    PROGRAM_GRAPH_FUNCTION_NODE_SHOULD_VALIDATE, PROGRAM_NAME, PROGRAM_ID, logger, MAX_FILES_NUMBER
from utils.files_utils import get_program_id_from_gexf_name


def plot_graph_data(graph_data):
    node_names = graph_data["demangled_function_names"][0]
    node_numbers = list(range(len(node_names)))
    edges_tensor = graph_data["edge_index"]
    edges_list = [(line[0].item(), line[1].item()) for line in edges_tensor.T]
    node_names = graph_data["demangled_function_names"][0]

    should_validate_mask = graph_data[PROGRAM_GRAPH_FUNCTION_NODE_SHOULD_VALIDATE]
    should_validate_mask = (should_validate_mask == 1).view(-1, 1)
    should_validate_numbers = [number for number, should_validate in zip(node_numbers, should_validate_mask) if
                               should_validate]

    logger.info(f'number of functions is {len(node_names)}')
    logger.info(f'number of named functions is {len(should_validate_mask)}')
    logger.info(f'number of edges is {len(edges_list)}')

    g = Graph()
    g.add_nodes_from(node_numbers)
    g.add_edges_from(edges_list)
    color_map = []
    for node in g:
        if node in should_validate_numbers:
            color_map.append('blue')
        else:
            color_map.append('green')
    labels = {number: name for number, name in zip(node_numbers, node_names)}
    plt.close()
    # networkx.draw(g, node_color=color_map, with_labels=True, labels=labels)
    # plt.show()


def print_and_save_graph(graph: Graph, file_path=None, is_print=True, is_save=False):
    function_names = networkx.get_node_attributes(graph, 'label')
    colors = [graph[u][v]['color'] for u, v in graph.edges()]
    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    ax.set_title('Random graph')
    networkx.draw(graph, labels=function_names, edge_color=colors, ax=ax)
    _ = ax.axis('off')
    if file_path is not None:
        plt.title(str(file_path.split('\\')[-1]))

    if is_save and file_path is not None:
        print('saving graph png')
        png_file_path = file_path + '.png'
        logger.info('png file has been saved to {}\n'.format(png_file_path))
        plt.savefig(png_file_path)  # save as png
    if is_print:
        print('showing graph')
        plt.show()


def from_mangled_name_to_demangled_name(function_mangled_name: str) -> str:
    name_to_demangle = function_mangled_name
    if function_mangled_name[0] == '.':
        name_to_demangle = name_to_demangle[1:]
    return cpp_demangle.demangle(name_to_demangle)


def remove_all_occurences_of_substring(full_string: str, sub_string: str) -> str:
    return full_string.replace(sub_string, '')


def from_demangled_name_to_name(function_demangled_name: str) -> str:
    index_of_sogar = function_demangled_name.find('(')
    if index_of_sogar != -1:
        function_demangled_name = function_demangled_name[:index_of_sogar]

    index_of_pointstaim = function_demangled_name.rfind('::')
    if index_of_pointstaim != -1:
        function_demangled_name = function_demangled_name[index_of_pointstaim + 2:]

    index_of_kavaim = function_demangled_name.rfind('__')
    if index_of_kavaim != -1:
        function_demangled_name = function_demangled_name[:index_of_kavaim] + function_demangled_name[
                                                                              index_of_kavaim + 2:]

    index_of_point = function_demangled_name.rfind('.')
    if index_of_point != -1:
        function_demangled_name = function_demangled_name[:index_of_point] + \
                                  function_demangled_name[index_of_point + 1:]

    index_of_template_bigger = function_demangled_name.rfind(' <')
    if index_of_template_bigger != -1:
        function_demangled_name = function_demangled_name[:index_of_template_bigger]

    function_demangled_name = remove_all_occurences_of_substring(function_demangled_name, '_')
    function_demangled_name = remove_all_occurences_of_substring(function_demangled_name, '__')
    function_demangled_name = remove_all_occurences_of_substring(function_demangled_name, '::')
    # print(function_demangled_name)
    return function_demangled_name


def transform_names_dict(original_names_dict: Dict, transform_function: callable):
    function_demangled_names = {}
    number_succeeded = 0
    for key, value in original_names_dict.items():
        try:
            function_demangled_names[key] = transform_function(value)
            number_succeeded += 1
        except:
            function_demangled_names[key] = value
    print(f'succeded in {str(transform_function)} in {number_succeeded} out of {len(original_names_dict)}')
    return function_demangled_names


def demangle_function_names_in_graph_labels(graph: networkx.Graph):
    function_names_mangled = networkx.get_node_attributes(graph, 'label')
    # TODO: try using additional demangling methods (not only cpp, maybe also c demangling?)
    function_demangled_names = transform_names_dict(function_names_mangled, from_mangled_name_to_demangled_name)
    function_real_names = transform_names_dict(function_demangled_names, from_demangled_name_to_name)

    networkx.set_node_attributes(graph, function_names_mangled, "mangled_function_names")
    networkx.set_node_attributes(graph, function_demangled_names, "demangled_function_names")
    networkx.set_node_attributes(graph, function_real_names, "label")

    print(f'{np.array([1 if real_name[:3] == "sub" else 0 for real_name in function_real_names.values()]).sum()} '
          f'functions that name starts with sub, out of {len(function_real_names)}')
    return graph


def add_addresses_to_graph(graph: Graph, childs_pickle_path):
    nodes_names = networkx.get_node_attributes(graph, 'label')


def add_edges_to_graph(graph: Graph, childs_pickle_path):
    mangled_nodes_names = networkx.get_node_attributes(graph, 'mangled_function_names')

    with open(childs_pickle_path, 'rb') as handle:
        call_address_to_function_variables = pickle.load(handle)

    function_address_to_function_node_id = {v: k for k, v in mangled_nodes_names.items()}
    edges_property_dict = {}
    edges_succ_type_list = {}
    edges_succ_address_list = {}
    colors = {}
    for call_nodes_per_function in call_address_to_function_variables.values():
        for call_json_str in call_nodes_per_function:
            if call_json_str is not None:
                call_json_unstr = json.loads(call_json_str)
                calling_function_name = call_json_unstr['calling_function_name']
                called_function_name = call_json_unstr['called_function_name']
                # TODO: be sure that it is ok that there are edges that aren't of things that exist
                if calling_function_name in function_address_to_function_node_id and called_function_name in function_address_to_function_node_id:
                    calling_function_node_id = function_address_to_function_node_id[calling_function_name]
                    called_function_node_id = function_address_to_function_node_id[called_function_name]
                    edge_name = (str(calling_function_node_id), str(called_function_node_id))
                    edges_property_dict[edge_name] = call_json_unstr
                    call_params = call_json_unstr['y']  # call_json_unstr['generic_successors']
                    call_return_val = call_json_unstr['x']
                    edges_succ_type_list[edge_name] = str([succ['type'] for succ in call_params])
                    edges_succ_address_list[edge_name] = str([succ['address'] for succ in call_params])
                    colors[edge_name] = 'red'
                else:
                    print('The caller or the called is not identified ( probably was down during strip -R .dysym)')

            else:
                print('The calling of the function has no object')
    networkx.set_edge_attributes(graph, edges_property_dict, name='call_atts')
    networkx.set_edge_attributes(graph, edges_succ_type_list, name='call_params_types')
    networkx.set_edge_attributes(graph, edges_succ_address_list, name='call_params_addresses')
    networkx.set_edge_attributes(graph, colors, name='color')

    return graph


def print_children_of_functions(childs_pickle_path):
    func_num = 0
    with open(childs_pickle_path, 'rb') as handle:
        call_address_to_function_variables = pickle.load(handle)
    print(str(len(call_address_to_function_variables.values())) + ' functions in total')
    for call_nodes_per_function in call_address_to_function_variables.values():
        func_num += 1
        print('function number ' + str(func_num) + ':')
        call_node_num = 0
        for call_json_str in call_nodes_per_function:
            call_node_num += 1
            print('call number' + str(call_node_num) + ':')
            if call_json_str is not None:
                call_json_unstr = json.loads(call_json_str)
                node_succs = call_json_unstr['y']
                print(str(len(node_succs)) + ' params for call.')
                print('param types: ' + str([succ['type'] for succ in node_succs]))
                print('param addresses: ' + str([succ['address'] for succ in node_succs]))
            else:
                print('None')


def get_node_id_by_mangled_name(graph, mangled_name):
    for node_id, function_mangled_name in networkx.get_node_attributes(graph, 'mangled_function_names').items():
        if mangled_name == function_mangled_name:
            return node_id
    print('Function {} doesn\'t have node'.format(mangled_name))
    # assert False, 'Function {} doesn\'t have node'.format(mangled_name)


def get_demangled_node_name_by_node_id(graph, node_id):
    for curr_node_id, function_demangled_name in \
            networkx.get_node_attributes(graph, PROGRAM_GRAPH_FUNCTION_NODE_DEMANGLED_NAME).items():
        if curr_node_id == node_id:
            return function_demangled_name
    return 'FAILED_ON_DEMANGLING - None'
    assert False, 'Failed to find demangled name for function with node id {}'.format(node_id)


class FileProcessedData:
    NotImplemented


class MergeFunctionsDataIntoNetworkx():
    def __init__(self, programs_data_path, networkx_programs_data_path, load_only_graphs=False):
        print("Merging data from function call graph into dire AST data, and moving files into raw folder...")
        self.examples = []
        self.programs_data_path = programs_data_path

        self.networkx_programs_data_path_stripped = Path(networkx_programs_data_path).joinpath('stripped')
        self.networkx_programs_data_path_stripped.mkdir(parents=True, exist_ok=True)

        self.networkx_programs_data_path_original = Path(networkx_programs_data_path).joinpath('original')
        self.networkx_programs_data_path_original.mkdir(parents=True, exist_ok=True)

        self.load_only_graphs = load_only_graphs

        if not self.load_only_graphs:
            self.process_and_save_to_raw_dataset_path()
        else:
            print('load only graphs')
            self.process_and_save_only_graphs_to_raw_dataset_path()

    def process_and_save_only_graphs_to_raw_dataset_path(self):
        pass

    def process_and_save_to_raw_dataset_path(self):
        stripped_functions_features_jsonl_path = os.path.join(self.programs_data_path,
                                                              STRIPPED_FUNCTIONS_FEATURES_FOLDER_NAME)
        original_functions_features_jsonl_path = os.path.join(self.programs_data_path,
                                                              ORIGINAL_FUNCTIONS_FEATURES_FOLDER_NAME)
        i = 0
        for stripped_function_features_file_name in os.listdir(
                stripped_functions_features_jsonl_path):  ##1200-1350 (1500) fault, 1750-1850 fault, 2400-2450, 2900- 3000 fault 3600, 3700 fault
            #TRAIN
            if i in ([0] + list(range(1200, 1350)) + list(range(1750, 1850)) + list(range(2400, 2450)) + list(
                    range(2900, 3000)) + list(range(3600, 3700)) + list(range(3900, 5000))):
                i = i + 1
                continue
            i = i + 1
            # Check that function features file is FUNCTIONS_FEATURES_FILE_EXTENSION ('jsonl')
            if os.path.splitext(stripped_function_features_file_name)[1] != DIRE_PROGRAM_DATA_FILE_EXTENSION:
                continue
            original_function_features_file_name = ''.join(stripped_function_features_file_name.split('.stripped'))
            if not os.path.exists(
                    os.path.join(original_functions_features_jsonl_path, original_function_features_file_name)):
                continue

            function_call_graphs_dir = os.path.join(self.programs_data_path, FUNCTION_CALL_GRAPHS_DIRECTORY)
            for stripped_function_call_graphs in os.listdir(function_call_graphs_dir):
                # Check that function features file is FUNCTION_CALL_GRAPHS_FILE_EXTENSION ('gexf')
                if os.path.splitext(stripped_function_call_graphs)[1] != PROGRAM_CALL_GRAPHS_FILE_EXTENSION:
                    continue

                original_function_call_graphs = ''.join(stripped_function_call_graphs.split('.stripped'))
                if not os.path.exists(
                        os.path.join(function_call_graphs_dir, original_function_call_graphs)):
                    continue

                # Check that the file names are the same
                if os.path.splitext(stripped_function_call_graphs)[0] == \
                        os.path.splitext(stripped_function_features_file_name)[0]:
                    # Create ProgramSources from the found files
                    non_stripped_function_call_graphs = '.'.join(
                        os.path.splitext(stripped_function_call_graphs)[0].split('.')[:-1]) + \
                                                        os.path.splitext(stripped_function_call_graphs)[1]

                    stripped_function_features_file_path = os.path.join(stripped_functions_features_jsonl_path,
                                                                        stripped_function_features_file_name)
                    original_function_features_file_path = os.path.join(original_functions_features_jsonl_path,
                                                                        original_function_features_file_name)

                    stripped_function_call_graphs_full_path = os.path.join(function_call_graphs_dir,
                                                                           stripped_function_call_graphs)

                    original_function_call_graphs_full_path = os.path.join(function_call_graphs_dir,
                                                                           original_function_call_graphs)

                    address_to_pickle_original_graph_full_path = os.path.join(function_call_graphs_dir,
                                                                              os.path.splitext(
                                                                                  non_stripped_function_call_graphs)[
                                                                                  0] + GRAPH_ADDR_TO_FUNC_NAME_FILENAME_SUFFIX)

                    original_function_call_graphs_full_path_old = os.path.join(function_call_graphs_dir,
                                                                               non_stripped_function_call_graphs)

                    assert original_function_call_graphs_full_path_old == original_function_call_graphs_full_path

                    address_to_pickle_stripped_graph_full_path = os.path.join(function_call_graphs_dir,
                                                                              os.path.splitext(
                                                                                  stripped_function_call_graphs)[
                                                                                  0] + GRAPH_ADDR_TO_FUNC_NAME_FILENAME_SUFFIX)

                    print(
                        'Creating example for program with files:\n\t{}\n\t{}'.format(
                            stripped_function_features_file_path,
                            stripped_function_call_graphs_full_path))

                    try:
                        stripped_program_file_name = os.path.basename(stripped_function_call_graphs_full_path)
                        stripped_graph = networkx.read_gexf(stripped_function_call_graphs_full_path)

                        original_program_file_name = os.path.basename(original_function_call_graphs_full_path)
                        original_graph = networkx.read_gexf(original_function_call_graphs_full_path)
                        original_graph.graph[PROGRAM_NAME] = original_function_features_file_name
                        original_graph.graph[PROGRAM_ID] = get_program_id_from_gexf_name(
                            original_function_features_file_name, original_functions_features_jsonl_path)

                        stripped_function_features_file_name = os.path.splitext(stripped_function_features_file_name)[0]
                        stripped_graph.graph[PROGRAM_NAME] = stripped_function_features_file_name
                        stripped_graph.graph[PROGRAM_ID] = original_graph.graph[PROGRAM_ID]
                        if not os.path.exists(address_to_pickle_stripped_graph_full_path):
                            continue
                        if not os.path.exists(address_to_pickle_original_graph_full_path):
                            continue

                        with open(address_to_pickle_stripped_graph_full_path,
                                  'rb') as f_address_to_pickle_stripped_graph_full_path:
                            stripped_addr_to_name = pickle.load(f_address_to_pickle_stripped_graph_full_path)

                        with open(address_to_pickle_original_graph_full_path,
                                  'rb') as f_address_to_pickle_original_graph_full_path:
                            original_addr_to_name = pickle.load(f_address_to_pickle_original_graph_full_path)

                        new_stripped_json_path = self.networkx_programs_data_path_stripped.joinpath(
                            os.path.splitext(stripped_program_file_name)[0] + DIRE_PROGRAM_DATA_FILE_EXTENSION)

                        new_original_json_path = self.networkx_programs_data_path_original.joinpath(
                            os.path.splitext(original_program_file_name)[0] + DIRE_PROGRAM_DATA_FILE_EXTENSION)
                        self.write_json_with_node_id(original_graph, original_graph,
                                                     original_function_features_file_path,
                                                     new_original_json_path)

                        self.write_json_with_node_id(original_graph, stripped_graph,
                                                     stripped_function_features_file_path,
                                                     new_stripped_json_path)
                    except:
                        continue

                    program_path = Path(self.networkx_programs_data_path_original).joinpath(original_program_file_name)
                    networkx.write_gexf(original_graph, program_path, prettyprint=False)

                    # for each mangled stripped name, find the matching mangled original name & the DEMANGLED original name.
                    stripped_name_to_original_demangled_name = {}
                    for stripped_addr, stripped_name in stripped_addr_to_name.items():
                        for original_addr, original_name in original_addr_to_name.items():
                            if stripped_addr == original_addr:
                                # Get demangled name, because these names are mangled
                                for node_key, node_data in original_graph.nodes.data():
                                    if node_data[PROGRAM_GRAPH_FUNCTION_NODE_MANGLED_NAME] == original_name:
                                        stripped_name_to_original_demangled_name[stripped_name] = node_data[
                                            PROGRAM_GRAPH_FUNCTION_NODE_DEMANGLED_NAME]
                                        break
                                break

                    # Add demangled original names to stripped and original graph
                    for node_info in stripped_graph.nodes.data():
                        node_data = node_info[1]
                        func_mangled_name = node_data[PROGRAM_GRAPH_FUNCTION_NODE_MANGLED_NAME]
                        if func_mangled_name in stripped_name_to_original_demangled_name.keys():
                            node_data[PROGRAM_GRAPH_FUNCTION_NODE_DEMANGLED_NAME] = \
                                stripped_name_to_original_demangled_name[func_mangled_name]

                    for node_info in stripped_graph.nodes.data():
                        node_data = node_info[1]
                        if PROGRAM_GRAPH_FUNCTION_NODE_DEMANGLED_NAME not in node_data.keys():
                            node_data[
                                PROGRAM_GRAPH_FUNCTION_NODE_DEMANGLED_NAME] = FUNCTION_DOESNT_APPEAR_IN_BOTH_GRAPHS_DEMANGLED_NAME_PLACEHOLDER

                    program_path = Path(self.networkx_programs_data_path_stripped).joinpath(stripped_program_file_name)
                    networkx.write_gexf(stripped_graph, program_path, prettyprint=False)
                    # Every function_features_file has exactly one function_call_graphs
                    break
            else:
                print('Didn\'t find call_graphs file for features file {}'.format(stripped_function_features_file_name))

    @staticmethod
    def write_json_with_node_id(original_graph, stripped_graph, functions_features_file, new_json_path):
        # Add the functions features to every node binary_search.o.jsonl
        with open(new_json_path, 'w+') as jsonl_file:
            with jsonlines.Writer(jsonl_file) as writer:
                for json_function_line in open(functions_features_file, 'r').readlines():
                    # info = process_single_function(ea)
                    json_dict = json.loads(json_function_line)
                    function_mangled_name = json_dict['function']
                    function_node_id = get_node_id_by_mangled_name(stripped_graph, function_mangled_name)
                    json_dict['node_id'] = function_node_id
                    json_dict[DEMANGLED_NAME] = get_demangled_node_name_by_node_id(stripped_graph, function_node_id)
                    writer.write(json_dict)
