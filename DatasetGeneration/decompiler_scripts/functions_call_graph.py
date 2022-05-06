
import json
import os
# from pathlib import Path
import pickle
import time
from collections import defaultdict

import idc
import ida_kernwin
import ida_xref
import idautils
import ida_funcs
import ida_hexrays

import re

from util import UNDEF_ADDR, FunctionGraph, GraphBuilder, hexrays_vars, get_expr_name, FunctionMetadata, get_function_arguments_names
import idaapi
import pprint

########### it is also in config. don't forget to change both!


FUNCTION_CALL_GRAPHS = 'function_call_graphs'


def get_call_node(graph_builder, call_ea):
    if graph_builder is None or call_ea is None:
        return None
    ea_nodes = []
    for node in graph_builder.function_graph.nodes:
        # Mostly there will be an `expr` node before the real `call` node which has the real successors.
        if node.hex_rays_item.ea == call_ea and node.hex_rays_item_type == 'call':
            return node
    # assert ea_nodes, 'Couldn\'t find node with type \'call\' for call in given address {}'.format(call_ea)
    return None


def get_ea_function_graph(ea):
    function_of_address = idaapi.get_func(ea)
    if function_of_address is None:
        return None, None
    try:
        decompiled_function = idaapi.decompile(function_of_address)
    except ida_hexrays.DecompilationFailure as e:
        function_name = ida_funcs.get_func_name(ea)
        print('Failed to decompile %x: %s!' % (ea, function_name))
        return None, None

    # Rename decompilation graph
    function_graph = FunctionGraph()
    graph_builder = GraphBuilder(function_graph)

    # Change graph builder according to decompiled_function.body
    graph_builder.apply_to(decompiled_function.body, None)
    function_name = ida_funcs.get_func_name(ea)

    return graph_builder, function_name


def get_function_addresses_to_names_dict():
    function_addresses = idautils.Functions()
    function_address_to_names_dict = {}
    for function_address in function_addresses:
        function_name = ida_funcs.get_func_name(function_address)
        function_address_to_names_dict[function_address] = function_name
    return function_address_to_names_dict


class FunctionsCallGraphBuilder:
    def activate(self):
        print('building graph vars.')
        graph_file_path = os.environ['GDL_FILE_DIR']
        function_addresses = idautils.Functions()

        cur = idc.MinEA()
        end = idc.MinEA()
        idc.GenCallGdl(graph_file_path, 'Call Gdl', idc.CHART_GEN_GDL)

        idc.Message('Gdl file has been saved to {}\n'.format(graph_file_path))
        print('functions call graph collected.')
        call_address_to_function_variables = self.addresses_to_calls_descriptions(function_addresses)
        childs_pickle_path = os.environ['CHILDREN_PICKLE_DIR']

        with open(childs_pickle_path, 'wb') as handle:
            pickle.dump(call_address_to_function_variables, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('dumped call_address_to_function_variables')

        function_address_to_names = get_function_addresses_to_names_dict()
        print('got function_address_to_names')

        function_address_to_names_path = os.path.join(os.environ['OUTPUT_DIR'], FUNCTION_CALL_GRAPHS,
                                                      os.environ['BINARY_NAME_PREFIX']) + '_addr_to_name.pkl'

        with open(function_address_to_names_path, 'wb') as handle:
            pickle.dump(function_address_to_names, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('dumped function_address_to_names')
        return 1

    def addresses_to_calls_descriptions(self, function_addresses):
        function_address_to_function_variables = {}
        for function_address in function_addresses:
            function_calls = list(idautils.CodeRefsTo(function_address, True))
            function_address_to_function_variables[function_address] = []
            for call in function_calls:
                graph_builder, calling_function_name = get_ea_function_graph(call)
                call_node = get_call_node(graph_builder, call)
                if call_node is not None:
                    call_node_json = call_node.to_stringable_json_object(graph_builder.function_graph.nodes)
                    call_node_json['calling_function_name'] = calling_function_name
                    call_node_json['called_function_name'] = ida_funcs.get_func_name(function_address)
                    call_node_json_str = json.dumps(call_node_json)
                    function_address_to_function_variables[function_address].append(call_node_json_str)
                else:
                    function_address_to_function_variables[function_address].append(None)
        return function_address_to_function_variables


def main():
    renamed_prefix = os.path.join(os.environ['OUTPUT_DIR'], 'functions',
                                  os.environ['BINARY_NAME_PREFIX'])
    # # Load collected variables
    # with open(os.environ['COLLECTED_VARS']) as vars_fh:
    #     varmap = pickle.load(vars_fh)

    # # run the graph builder
    call_graph_builder = FunctionsCallGraphBuilder()
    call_graph_builder.activate()


idaapi.autoWait()
if not idaapi.init_hexrays_plugin():
    idaapi.load_plugin('hexrays')
    idaapi.load_plugin('hexx64')
    if not idaapi.init_hexrays_plugin():
        print('Unable to load Hex-rays')
    else:
        print('Hex-rays version %s has been detected' % idaapi.get_hexrays_version())
main()
ida_pro.qexit(0)
