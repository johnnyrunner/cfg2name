import pprint
from collections import defaultdict

import ida_funcs
import ida_hexrays
import ida_lines
import ida_pro
import json
import re

import ida_xref
import idautils
import idc

import idaapi

NUM_PARAM_TYPE = 'NUM'
ARG_PARAM_TYPE = 'ARG'

UNDEF_ADDR = 0xFFFFFFFFFFFFFFFF
hexrays_vars = re.compile("^(v|a)[0-9]+$")


def get_expr_name(expr):
    name = expr.print1(None)
    name = ida_lines.tag_remove(name)
    name = ida_pro.str2user(name)
    return name


def is_allowed_char(char):
    return char == '_' or char.isalnum()


def get_param_actual_name(var_plus_type_name):
    reversed_name = ''
    for char in var_plus_type_name[::-1]:
        if is_allowed_char(char):
            reversed_name += char
        else:
            return reversed_name[::-1]


def get_function_arguments_names(function_address):
    # nodes[0] - function beginning
    cmt = idc.GetType(function_address)
    if cmt is not None:
        after_openning_sogar = cmt.split("(")
        before_ends_sogar = after_openning_sogar[1].split(")")
        var_plus_type_names = before_ends_sogar[0].split(",")
        return [get_param_actual_name(var_plus_type_name) for var_plus_type_name in var_plus_type_names]
    return None
    # chidren_names = [get_expr_name(succ.hex_rays_item.cexpr) for succ in nodes[0]['generic_successors']]:


class GraphNode(object):
    def __init__(self, hex_rays_item, node_id):
        # ids of successors
        self.successors = []
        # ids of predecessors
        self.predecessors = []
        self.hex_rays_item = hex_rays_item

        # node id is the index of GraphNOde in FunctionGraph's nodes.
        self.node_id = node_id
        self.hex_rays_item_type = ida_hexrays.get_ctype_name(self.hex_rays_item.op)
        self.var_name = None

        # If item is variable, add its var name
        if self.hex_rays_item.op is ida_hexrays.cot_var:
            self.var_name = get_expr_name(self.hex_rays_item)

            # TODO: Here we maybe want to change ea if its UNDEF_ADDR to the node's predecessor ea.
            # if self.hex_rays_item.ea != UNDEF_ADDR:
            #     self.var_addresses.append(self.hex_rays_item.ea)
            # Try getting the ea of the predecessor
            # else:
            #     item_id = self.function_graph.reverse[self.hex_rays_item]
            #     ea = self.function_graph.get_pred_ea(item_id)
            #     if ea != UNDEF_ADDR:
            #         self.addresses[name].add(ea)

    def to_json_object(self, nodes):
        """
        :return: information about the node
        """
        # Each node has a unique ID
        node_info = {"node_id": self.node_id}

        # This is the type of ctree node
        node_info["hex_rays_item_type"] = self.hex_rays_item_type

        # This is the type of the data (in C-land)
        if self.hex_rays_item.is_expr() and not self.hex_rays_item.cexpr.type.empty():
            node_info["type"] = self.hex_rays_item.cexpr.type._print()

        node_info["address"] = "%08X" % self.hex_rays_item.ea
        #
        # if item.ea == UNDEF_ADDR:
        #     node_info["parent_address"] = "%08X" % self.get_pred_ea(n)

        # Specific info for different node types
        if self.hex_rays_item.op == ida_hexrays.cot_ptr:
            node_info["pointer_size"] = self.hex_rays_item.cexpr.ptrsize
        elif self.hex_rays_item.op == ida_hexrays.cot_memptr:
            node_info.update({
                "pointer_size": self.hex_rays_item.cexpr.ptrsize,
                "m": self.hex_rays_item.cexpr.m
            })
        elif self.hex_rays_item.op == ida_hexrays.cot_memref:
            node_info["m"] = self.hex_rays_item.cexpr.m
        elif self.hex_rays_item.op == ida_hexrays.cot_obj:
            node_info.update({
                "name": get_expr_name(self.hex_rays_item.cexpr),
                "ref_width": self.hex_rays_item.cexpr.refwidth
            })
        elif self.hex_rays_item.op == ida_hexrays.cot_var:
            # Try splitting the cexpr
            try:
                _, var_id, old_name, new_name = get_expr_name(self.hex_rays_item.cexpr).split("@@")
                node_info.update({
                    "var_id": var_id,
                    "old_name": old_name,
                    "new_name": new_name,
                    "ref_width": self.hex_rays_item.cexpr.refwidth
                })
            # In case unpacking didn't succeed
            except ValueError:
                node_info['expr_name'] = self.hex_rays_item.cexpr

        elif self.hex_rays_item.op in [ida_hexrays.cot_num,
                                       ida_hexrays.cot_str,
                                       ida_hexrays.cot_helper]:
            node_info["name"] = get_expr_name(self.hex_rays_item.cexpr)
        # Get info for children of this node
        generic_successors = self.successors[:]
        try:
            node_info['name'] = get_expr_name(self.hex_rays_item.cexpr)
        except:
            pass
        # If the hex rays item is an expression, get info about the x, y & z, which exists only in expressions
        if self.hex_rays_item.is_expr():
            for sucessors_type in ['x', 'y', 'z']:
                for successor_id in generic_successors:
                    # start the comparison if with checking if this sucessor type is not None
                    if getattr(self.hex_rays_item, sucessors_type) and \
                            getattr(self.hex_rays_item, sucessors_type) == nodes[successor_id].hex_rays_item:
                        generic_successors.remove(successor_id)
                        node_info[sucessors_type] = nodes[successor_id].to_json_object(nodes)
                        break
        node_info["generic_successors"] = [nodes[remaining_generic_successor_id].to_json_object(nodes) for
                                           remaining_generic_successor_id in generic_successors]
        return node_info

    def to_stringable_json_object(self, nodes):
        # Each node has a unique ID
        node_info = {"node_id": self.node_id}

        # # This is the type of ctree node
        # node_info["hex_rays_item_type"] = self.hex_rays_item_type
        #
        # # This is the type of the data (in C-land)
        if self.hex_rays_item.is_expr() and not self.hex_rays_item.cexpr.type.empty():
            node_info["type"] = self.hex_rays_item.cexpr.type._print()

        node_info["address"] = "%08X" % self.hex_rays_item.ea

        try:
            my_name = get_expr_name(self.hex_rays_item.cexpr)
            node_info["name"] = my_name
        except:
            my_name = None

        argument_names = get_function_arguments_names(idaapi.get_func(nodes[0].hex_rays_item.ea).startEA)
        if my_name is not None and argument_names is not None and my_name in argument_names:
            node_info['parameter_type'] = ARG_PARAM_TYPE


        if self.hex_rays_item_type == 'num':
            if my_name is not None:
                node_info['parameter_type'] = NUM_PARAM_TYPE
            else:
                node_info['parameter_type'] = my_name

        #
        # if item.ea == UNDEF_ADDR:
        #     node_info["parent_address"] = "%08X" % self.get_pred_ea(n)

        # Specific info for different node types
        # if self.hex_rays_item.op == ida_hexrays.cot_ptr:
        #     node_info["pointer_size"] = self.hex_rays_item.cexpr.ptrsize
        # elif self.hex_rays_item.op == ida_hexrays.cot_memptr:
        #     node_info.update({
        #         "pointer_size": self.hex_rays_item.cexpr.ptrsize,
        #         "m": self.hex_rays_item.cexpr.m
        #     })
        # elif self.hex_rays_item.op == ida_hexrays.cot_memref:
        #     node_info["m"] = self.hex_rays_item.cexpr.m
        # elif self.hex_rays_item.op == ida_hexrays.cot_obj:
        #     node_info.update({
        #         "name": get_expr_name(self.hex_rays_item.cexpr),
        #         "ref_width": self.hex_rays_item.cexpr.refwidth
        #     })
        # elif self.hex_rays_item.op == ida_hexrays.cot_var:
        #     # Try splitting the cexpr
        #     try:
        #         _, var_id, old_name, new_name = get_expr_name(self.hex_rays_item.cexpr).split("@@")
        #         node_info.update({
        #             "var_id": var_id,
        #             "old_name": old_name,
        #             "new_name": new_name,
        #             "ref_width": self.hex_rays_item.cexpr.refwidth
        #         })
        #     # In case unpacking didn't succeed
        #     except ValueError:
        #         node_info['expr_name'] = self.hex_rays_item.cexpr
        #
        # elif self.hex_rays_item.op in [ida_hexrays.cot_num,
        #                  ida_hexrays.cot_str,
        #                  ida_hexrays.cot_helper]:
        #     node_info["name"] = get_expr_name(self.hex_rays_item.cexpr)
        # Get info for children of this node
        generic_successors = self.successors[:]

        # # If the hex rays item is an expression, get info about the x, y & z, which exists only in expressions
        if self.hex_rays_item.is_expr():
            for sucessors_type in ['x', 'y', 'z']:
                node_info[sucessors_type] = []
                for successor_id in generic_successors:
                    # start the comparison if with checking if this sucessor type is not None
                    if getattr(self.hex_rays_item, sucessors_type) and \
                            getattr(self.hex_rays_item, sucessors_type) == nodes[successor_id].hex_rays_item:
                        generic_successors.remove(successor_id)
                        node_info[sucessors_type] = nodes[successor_id].to_stringable_json_object(nodes)
                        break
        node_info["generic_successors"] = [nodes[remaining_generic_successor_id].to_stringable_json_object(nodes) for
                                           remaining_generic_successor_id in generic_successors]
        return node_info


class FunctionGraph(object):
    def __init__(self):
        # Our wrapping of self.items, contains GraphNode
        self.nodes = []

    def add_edge(self, predecessor, successor):
        # Add to our nodes.
        self.nodes[predecessor.node_id].successors.append(successor.node_id)
        self.nodes[successor.node_id].predecessors.append(predecessor.node_id)

    # def get_pred_ea(self, n):
    #     if self.npred(n) == 1:
    #         pred = self.pred(n, 0)
    #         pred_item = self.items[pred]
    #         if pred_item.ea == UNDEF_ADDR:
    #             return self.get_pred_ea(pred)
    #         return pred_item.ea
    #
    #     return UNDEF_ADDR
    #
    # def get_node_label(self, n):
    #     item = self.items[n]
    #     op = item.op
    #     insn = item.cinsn
    #     expr = item.cexpr
    #     parts = [ida_hexrays.get_ctype_name(op)]
    #     if op == ida_hexrays.cot_ptr:
    #         parts.append(".%d" % expr.ptrsize)
    #     elif op == ida_hexrays.cot_memptr:
    #         parts.append(".%d (m=%d)" % (expr.ptrsize, expr.m))
    #     elif op == ida_hexrays.cot_memref:
    #         parts.append(" (m=%d)" % (expr.m,))
    #     elif op in [
    #         ida_hexrays.cot_obj,
    #         ida_hexrays.cot_var]:
    #         name = get_expr_name(expr)
    #         parts.append(".%d %s" % (expr.refwidth, name))
    #     elif op in [
    #         ida_hexrays.cot_num,
    #         ida_hexrays.cot_helper,
    #         ida_hexrays.cot_str]:
    #         name = get_expr_name(expr)
    #         parts.append(" %s" % (name,))
    #     elif op == ida_hexrays.cit_goto:
    #         parts.append(" LABEL_%d" % insn.cgoto.label_num)
    #     elif op == ida_hexrays.cit_asm:
    #         parts.append("<asm statements; unsupported ATM>")
    #         # parts.append(" %a.%d" % ())
    #     parts.append(", ")
    #     parts.append("ea: %08X" % item.ea)
    #     if item.is_expr() and not expr is None and not expr.type.empty():
    #         parts.append(", ")
    #         tstr = expr.type._print()
    #         parts.append(tstr if tstr else "?")
    #     return "".join(parts)

    def get_full_tree(self):
        # full tree is the json object of the first node.
        print(json.dumps(self.nodes[0].to_json_object(self.nodes)))


class GraphBuilder(ida_hexrays.ctree_parentee_t):
    def __init__(self, function_graph):
        ida_hexrays.ctree_parentee_t.__init__(self)
        self.function_graph = function_graph

    def print_node(self, node_id):
        if node_id > len(self.function_graph.nodes):
            print('you dumv')
            assert False
        else:
            pprint.pprint(self.function_graph.nodes[node_id].to_json_object(self.function_graph.nodes))

    def add_node(self, new_hex_rays_item):
        # Add new node, with unique id which is the length of nodes.
        self.function_graph.nodes.append(
            GraphNode(new_hex_rays_item, len(self.function_graph.nodes))
        )

        # Return new node, which just got inserted.
        return self.function_graph.nodes[-1]

    def process(self, new_hex_rays_item):
        new_node = self.add_node(new_hex_rays_item)

        # VOODOO, but calls add_edge, which is what we want.
        # I think that self is injected every call with parents of new_hex_rays_item
        if len(self.parents) > 0:
            parent = self.parents.back()
            if parent is None:
                return 0

            for maybe_parent_to_new_node in self.function_graph.nodes:
                if type(maybe_parent_to_new_node.hex_rays_item.to_specific_type) != type(parent.to_specific_type):
                    continue

                if maybe_parent_to_new_node.hex_rays_item.to_specific_type == parent.to_specific_type:
                    self.function_graph.add_edge(maybe_parent_to_new_node, new_node)
                    break

    def visit_insn(self, instance):
        self.process(instance)

        # Original signature returns int.
        return 0

    def visit_expr(self, expression):
        self.process(expression)

        # Original signature returns int.
        return 0


class FunctionMetadata(object):
    def __init__(self, ea):
        """
        :param ea: ea must be a pointer to a function
        """

        # IDA internal types
        self.ea = ea
        self.func_t = ida_funcs.get_func(self.ea)

        # Metadata
        # self.size = self.func_t.size
        self.does_return = self.func_t.does_return()
        # TODO: why always true
        self.is_entry = ida_funcs.is_func_entry(self.func_t)
        self.name = ida_funcs.get_func_name(self.ea)
        self.code_refs_to_function = list(idautils.CodeRefsTo(ea, True))
        self.code_refs_from_function = list(idautils.CodeRefsFrom(ea, True))
        self.data_refs_to_function = list(idautils.DataRefsTo(ea))
        self.data_refs_from_function = list(idautils.DataRefsFrom(ea))
        self.xrefs_from_function = list(idautils.XrefsFrom(ea, ida_xref.XREF_ALL))
        self.xrefs_to_function = list(idautils.XrefsTo(ea, ida_xref.XREF_ALL))

# TODO: Use GraphNode's self.var_addresses to replace these classes. They are meant to convert names of variables to
# generic consistent names (var_a will be var_a in all its nodes, not VAR_A@@VAR_).
# class AddressCollector:
#     def __init__(self, cg):
#         self.cg = cg
#         self.addresses = defaultdict(set)
#
#     def collect(self):
#         for item in self.cg.items:
#             if item.op is ida_hexrays.cot_var:
#                 name = get_expr_name(item)
#                 if item.ea != UNDEF_ADDR:
#                     self.addresses[name].add(item.ea)
#                 else:
#                     item_id = self.cg.reverse[item]
#                     ea = self.cg.get_pred_ea(item_id)
#                     if ea != UNDEF_ADDR:
#                         self.addresses[name].add(ea)
#
# class RenamedGraphBuilder(GraphBuilder):
#     def __init__(self, cg, func, addresses):
#         self.func = func
#         self.addresses = addresses
#         super(RenamedGraphBuilder, self).__init__(cg)
#
#     def visit_expr(self, e):
#         global var_id
#         if e.op is ida_hexrays.cot_var:
#             # Save original name of variable
#             original_name = get_expr_name(e)
#             sentinel_vars = re.compile('@@VAR_[0-9]+')
#             if not sentinel_vars.match(original_name):
#                 # Get new name of variable
#                 addresses = frozenset(self.addresses[original_name])
#                 if addresses in varmap and varmap[addresses] != '::NONE::':
#                     new_name = varmap[addresses]
#                 else:
#                     new_name = original_name
#                 # Save names
#                 varnames[var_id] = (original_name, new_name)
#                 # Rename variables to @@VAR_[id]@@[orig name]@@[new name]
#                 self.func.get_lvars()[e.v.idx].name = '@@VAR_' + str(var_id) + '@@' + original_name + '@@' + new_name
#                 var_id += 1
#         return self.process(e)
