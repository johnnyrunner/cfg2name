from typing import List, Dict
import torch
import Datasets.dire_raw_dataset
from config import *
from dire_neural_model.utils import nn_util
from dire_neural_model.utils.dire_vocab import DireVocab, SAME_VARIABLE_TOKEN
from Datasets.dire_raw_dataset import *
from dire_neural_model.utils.grammar import Grammar
from vocabulary_builder.dire_variables_vocab import DireVariablesVocab


class DireTensorDataset:
    @staticmethod
    def functions_data_to_tensor_dict(functions_data: List[DireRawFunctionData], vocab: DireVariablesVocab, load_nero_vectors):
        dire_tensor_dataset = DireTensorDataset(vocab.dire_vocab)
        return dire_tensor_dataset.functions_data_to_tensor_dict_imp(functions_data, load_nero_vectors)

    @property
    def vocab(self) -> DireVocab:
        return self._vocab

    @property
    def grammar(self) -> Grammar:
        return self._grammar

    def __init__(self, vocab: DireVocab):
        self._vocab = vocab
        self._grammar = vocab.grammar

    def fill_function_data_with_vocab_data(self, function_data) -> DireRawFunctionData:
        """annotate examples by populating specific fields, useful for sorting examples or batching"""
        # for ensemble models, it will be annotated by the batcher for each specific class

        src_bpe_model = self.vocab.source_tokens.subtoken_model
        snippet = function_data.code_tokens
        snippet = ' '.join(snippet)
        sub_tokens = ['<s>'] + src_bpe_model.encode_as_pieces(snippet) + ['</s>']
        sub_token_ids = [src_bpe_model.bos_id()] + src_bpe_model.encode_as_ids(snippet) + [src_bpe_model.eos_id()]
        setattr(function_data, 'sub_tokens', sub_tokens)
        setattr(function_data, 'sub_token_ids', sub_token_ids)
        setattr(function_data, 'source_seq_length', len(sub_tokens))

        tgt_bpe_model = self.vocab.target.subtoken_model
        eov_id = tgt_bpe_model.eos_id()
        variable_name_subtoken_map = dict()
        tgt_pred_seq_len = 0
        for old_name, new_name in function_data.variable_name_map.items():
            if old_name == new_name:
                subtoken_ids = [self.vocab.target[SAME_VARIABLE_TOKEN], eov_id]
            else:
                subtoken_ids = tgt_bpe_model.encode_as_ids(new_name) + [eov_id]
            variable_name_subtoken_map[old_name] = subtoken_ids
            tgt_pred_seq_len += len(subtoken_ids)

        setattr(function_data, 'variable_name_subtoken_map', variable_name_subtoken_map)
        setattr(function_data, 'target_prediction_seq_length', tgt_pred_seq_len)

        return function_data

    def functions_data_to_prediction_target(self, functions_data: List[DireRawFunctionData]):
        batch_size = len(functions_data)
        unchanged_var_weight = UNCHANGED_VARIABLE_WEIGHT
        variable_name_subtoken_maps = [e.variable_name_subtoken_map for e in functions_data]
        asts = [e.ast for e in functions_data]
        var_name_maps = [e.variable_name_map for e in functions_data]
        examples_tensors_dict = self.extract_functions_data_prediction_tensors(batch_size, asts, var_name_maps,
                                                                               unchanged_var_weight,
                                                                               variable_name_subtoken_maps)
        return examples_tensors_dict

    @staticmethod
    def extract_functions_data_prediction_tensors(batch_size, asts, var_name_maps, unchanged_var_weight,
                                                  variable_name_subtoken_maps):
        # Batch size = amount of asts
        max_pred_timestep = max(sum(len(val) for val in x.values()) for x in variable_name_subtoken_maps)
        target_variable_encoding_indices = torch.zeros(batch_size, max_pred_timestep, dtype=torch.long)
        target_variable_encoding_indices_mask = torch.zeros(batch_size, max_pred_timestep)
        variable_tgt_name_id = torch.zeros(batch_size, max_pred_timestep, dtype=torch.long)
        variable_tgt_name_weight = torch.zeros(batch_size, max_pred_timestep)
        var_with_new_name_mask = torch.zeros(batch_size, max_pred_timestep)
        auxiliary_var_mask = torch.zeros(batch_size, max_pred_timestep)
        variable_master_node_ptr = 0
        for e_id, (ast, var_name_map) in enumerate(zip(asts, var_name_maps)):
            _var_node_ids = []
            _tgt_name_ids = []
            variable_ptr = 0
            for var_id, var_name in enumerate(ast.variables):
                new_var_name_subtoken_ids = variable_name_subtoken_maps[e_id][var_name]
                variable_end_ptr = variable_ptr + len(new_var_name_subtoken_ids)

                variable_tgt_name_id[e_id, variable_ptr: variable_end_ptr] = torch.tensor(new_var_name_subtoken_ids,
                                                                                          dtype=torch.long)

                if var_name == var_name_map[var_name]:
                    auxiliary_var_mask[e_id, variable_ptr: variable_end_ptr] = 1.
                    variable_tgt_name_weight[e_id, variable_ptr: variable_end_ptr] = unchanged_var_weight
                else:
                    var_with_new_name_mask[e_id, variable_ptr: variable_end_ptr] = 1.
                    variable_tgt_name_weight[e_id, variable_ptr: variable_end_ptr] = 1.

                target_variable_encoding_indices[e_id,
                variable_ptr: variable_end_ptr] = var_id  # variable_master_node_ptr

                variable_master_node_ptr += 1
                variable_ptr = variable_end_ptr

            target_variable_encoding_indices_mask[e_id, :variable_ptr] = 1.

        # method_tgt_name_ids = torch.zeros(batch_size, max(len(method_names_subtokens)), dtype=torch.long)
        # for e_id, method_name_subtokens in enumerate(method_names_subtokens):
        #     method_tgt_name_ids[e_id, 0: len(method_names_subtokens)] = torch.tensor(method_names_subtokens,
        #                                                                           dtype=torch.long)
        # TODO: add method name weights (to ignore non-interesting or fake methods like trampolines)

        examples_tensors_dict = dict(variable_tgt_name_id=variable_tgt_name_id,
                                     variable_tgt_name_weight=variable_tgt_name_weight,
                                     var_with_new_name_mask=var_with_new_name_mask,
                                     auxiliary_var_mask=auxiliary_var_mask,
                                     target_variable_encoding_indices=target_variable_encoding_indices,
                                     target_variable_encoding_indices_mask=target_variable_encoding_indices_mask)
        return examples_tensors_dict

    def get_asts_variables(self, asts):
        var_ids = []
        var_names = []
        for ast in asts:
            var_ids_for_ast = []
            var_names_for_ast = []
            for var_id, var_name in enumerate(ast.variables):
                var_ids_for_ast.append(var_id)
                var_names_for_ast.append(var_name)
            var_ids.append(var_ids_for_ast)
            var_names.append(var_names_for_ast)

        asts_variables = dict(
            # var_ids=torch.tensor(var_ids),
            var_names=var_names
        )
        return asts_variables

    def get_asts_variable_name_map(self, variable_name_maps):
        all_old_names = []
        all_gold_new_names = []
        for variable_name_map in variable_name_maps:
            old_names = []
            gold_new_names = []
            for old_name, gold_new_name in variable_name_map.items():
                old_names.append(old_name)
                gold_new_names.append(gold_new_name)
            all_old_names.append(old_names)
            all_gold_new_names.append(gold_new_names)

        _variable_name_maps = dict(
            all_old_names=all_old_names,
            all_gold_new_names=all_gold_new_names
        )
        return _variable_name_maps


    def functions_data_to_tensor_dict_imp(self, functions_data: List[DireRawFunctionData], load_nero_vectors) \
            -> Dict[str, torch.Tensor]:

        from dire_neural_model.model.graph_encoder import GraphASTEncoder
        from dire_neural_model.model.sequential_encoder import SequentialEncoder

        for function_data in functions_data:
            self.fill_function_data_with_vocab_data(function_data)

        asts = [e.ast for e in functions_data]
        packed_graph, gnn_tensor_dict = GraphASTEncoder.to_packed_graph(asts,
                                                                        connections=DIRE_AST_GRAPH_CONNECTIONS)
        gnn_tensors = GraphASTEncoder.to_tensor_dict(packed_graph, self.grammar, self.vocab)
        gnn_tensor_dict.update(gnn_tensors)

        seq_tensor_dict = SequentialEncoder.to_tensor_dict(functions_data)

        tensor_dict = {'graph_encoder_input': gnn_tensor_dict,
                       'seq_encoder_input': seq_tensor_dict}

        prediction_target = self.functions_data_to_prediction_target(functions_data)
        tensor_dict['prediction_target'] = prediction_target

        asts_variables = self.get_asts_variables(asts)
        tensor_dict['asts_variables'] = asts_variables
        if load_nero_vectors:
            tensor_dict['nero_id'] = [function_data.nero_id for function_data in functions_data]

        variable_name_maps = [e._variable_name_map for e in functions_data]
        _variable_name_maps = self.get_asts_variable_name_map(variable_name_maps)
        tensor_dict['_variable_name_map'] = _variable_name_maps
        # SP: wtf is this
        if hasattr(functions_data[0], 'test_meta'):
            tensor_dict['test_meta'] = [e.test_meta for e in functions_data]

        tensor_dict['batch_size'] = len(functions_data)
        num_elements = nn_util.get_tensor_dict_size(tensor_dict)
        tensor_dict['num_elements'] = num_elements
        return tensor_dict
