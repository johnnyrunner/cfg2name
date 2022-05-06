import copy
import json
from itertools import compress

import networkx
import torch_geometric
from torch_geometric.utils import from_networkx, subgraph
from tqdm import tqdm

from Datasets.dire_cache import dire_cache
from Datasets.graph_features_builder import get_mangled_and_demangled_names_from_networkx_graph
from Datasets.nero_cache import nero_cache
from config import *
from torch_geometric.data import Dataset, Data

from Datasets.dire_raw_dataset import DireRawSingleProgramDataset, List
from Datasets.dire_tensor_dataset import DireTensorDataset
from dire_neural_model.utils import nn_util
from dire_neural_model.utils.ast import AbstractSyntaxTree
from dire_neural_model.utils.evaluation import Evaluator
from utils.files_utils import inplace_change_word_occurences_in_file, delete_dataset_if_needed
from utils.general_utils import get_masked_iterable
from utils.graph_utils import MergeFunctionsDataIntoNetworkx
from vocabulary_builder.full_vocab import FullVocab
import os.path as osp

from vocabulary_builder.functions_vocab import FunctionsVocab

import pickle

SMALL_PROGRAM_SIZE = 100
counter_out = 0
counter = 0

counter_point = 0


def function_name_starts_with_sub(function_name: str):
    return function_name[:3] == 'sub'


def remove_broken_file(raw_path, processed_path):
    print(f'file in {raw_path} is bad')
    print('removing broken file')
    os.remove(raw_path)
    try:
        os.remove(processed_path)
    except OSError:
        pass


def get_name_from_path(raw_path):
    splitted_raw_path = raw_path.split('__')
    lib = splitted_raw_path[-1].split('.')[0]
    exe = splitted_raw_path[-2]
    to_remove = '1234567890.-'
    if exe[0] == '.':
        exe = exe[0] + ''.join([c for c in exe[1:] if c not in to_remove])
    else:
        exe = ''.join([c for c in exe if c not in to_remove])
    return f'{lib}@{exe}'


def get_nero_name_from_name(name):
    return name.replace('_', '*')


def is_example_bad(example):
    return dire_cache.is_hash_bad(example)


class ProgramsDataset(Dataset):
    def __init__(self,
                 dataset_dir: DatasetDirs,
                 load_dire_vocab_from_file: bool,
                 dire_vocab_file=DIRE_VOCABULARY_EXAMPLE_FILE,
                 functions_object_dir=None,  # FUNCTIONS_VOCABULARY_EXAMPLE_FILE
                 vocab_pickle_file_name=VOCAB_PICKLE_FILE_NAME,
                 force_remove_names_from_dire: bool = True,
                 stripped_or_original: str = ORIGINAL,
                 functions_vocab_size: int = 1000,
                 only_small_dataset: bool = False,
                 small_program_size=SMALL_PROGRAM_SIZE,
                 load_vocab_from_dataset=None,
                 size_of_subsamples=-1,
                 number_of_subsamples=3,
                 use_subgraphs_method=False,
                 retrain_functions_vocab=False,
                 load_nero_vectors=False,  # TODO: add this as True to all those that experiments need nero
                 *args):
        self.load_nero_vectors = load_nero_vectors
        self.only_small_dataset = only_small_dataset
        self.small_program_size = small_program_size
        if load_dire_vocab_from_file:
            print('this is the place vocab is init when dire vocab is loaded and other vocab is generated')
            if functions_object_dir is None:
                functions_object_dir = os.path.join(dataset_dir, 'functions_vocab_dir')
            self._vocab = FullVocab(load_dire_from_files=True, dire_vocab_file=dire_vocab_file,
                                    functions_vocab_file=functions_object_dir)
        else:
            print('dire and functions vocab will be built when dataset.process() function will be built')
        self.force_remove_function_names_from_dire = force_remove_names_from_dire

        assert stripped_or_original in [STRIPPED, ORIGINAL]
        self.stripped_or_original = stripped_or_original

        if load_vocab_from_dataset is not None:
            self._vocab = load_vocab_from_dataset._vocab
            self.functions_vocab_size = load_vocab_from_dataset.functions_vocab_size
        else:
            self.functions_vocab_size = functions_vocab_size

        self.vocab_dir = osp.join(dataset_dir, vocab_pickle_file_name)
        self.size_of_subsamples = size_of_subsamples
        self.number_of_subsamples = number_of_subsamples
        self.num_examples_per_processed = {}
        self.use_subgraphs_method = use_subgraphs_method
        super(ProgramsDataset, self).__init__(dataset_dir, *args)
        self._vocab = torch.load(self.vocab_dir)
        self.retrain_functions_vocab = retrain_functions_vocab
        if retrain_functions_vocab:
            self.init_functions_vocab()

        self._examples = []
        self._all_function_names = []
        self.load_data_to_memory()

    def change_to_new_vocab(self, data_graph):
        for node_info in data_graph.nodes.data():
            node_data = node_info[1]
            demangled_name = node_data[PROGRAM_GRAPH_FUNCTION_NODE_DEMANGLED_NAME]
            node_data[PROGRAM_GRAPH_FUNCTION_NODE_DEMANGLED_NAME_TOKENIZED] = \
                self._vocab.functions_vocab.encode_as_subtoken_ids(demangled_name)

    def examples_siever(self):
        print('siever')
        hashes_inserted = {}
        examples = [example for example in self._examples if not is_example_bad(example)]
        self._examples = []
        for example in tqdm(examples):
            hash_example = dire_cache.hash_example(example, i=0)
            if hash_example not in hashes_inserted:
                print(f'hash example: {hash_example}')
                self._examples.append(example)
                hashes_inserted[hash_example] = True
            else:
                print('already inside')

    def load_data_to_memory(self):
        evaluator = Evaluator()
        for processed_path in tqdm(self.processed_paths, unit='processed path'):
            try:
                data_graphs, data_asts, num_examples_per_processed = torch.load(processed_path)
                self.num_examples_per_processed[processed_path] = num_examples_per_processed
                for data_graph, data_ast in zip(data_graphs, data_asts):
                    if self.retrain_functions_vocab:
                        # data_asts = self.change_to_new_vocab(data_asts)
                        print('probably fault due to change in vocab')
                    self._examples.append((data_ast, data_graph))
                    concat_seperated_words_in_y, concat_words_in_y, set_words_in_ys, words_id_in_y_unique = evaluator.get_words_from_ones_tensor(
                        self._vocab.functions_vocab.get_id2word(),
                        data_ast[PROGRAM_GRAPH_FUNCTION_NODE_DEMANGLED_NAME_TOKENIZED])
                    sets_words_in_y = [frozenset(words_in_function) for words_in_function in set_words_in_ys]
                    self._all_function_names += sets_words_in_y
            except:
                print('file not found')
                # raise ()
        # self._len = sum([self.num_examples_per_processed[processed_path] for processed_path in self.processed_paths])
        self.examples_siever()
        self._len = len(self._examples)

    # @staticmethod
    # def data_graph_of_nodes_subset(data_graph, nodes_mask):
    #     print(data_graph)
    #     new_data_graph = Data()
    #     edge_index_subgraph, edges_mask = subgraph(nodes_mask, data_graph['edge_index'],
    #                                                edge_attr=torch.IntTensor([int(x) for x in data_graph['id']]))
    #     nodes_atts = {'align': True, 'arrowstyle': False,
    #                   'bordercolor': True, 'call_atts': False,
    #                   'call_succs_addresses': False, 'call_succs_types': False,
    #                   'color': False, 'demangled_function_name_tokenized': True,
    #                   'demangled_function_names': True, 'edge_index': False,
    #                   'fill': True, 'function type': False,
    #                   'function_address': True, 'id': False, 'label': True,
    #                   'mangled_function_names': True, 'num_params': False,
    #                   'num_params_ida': False, 'return address': False,
    #                   'return type': False, 'segment_name': True,
    #                   'should_validate': True}
    #     for key, value in data_graph:
    #         if nodes_atts[key]:
    #             new_data_graph[key] = get_masked_iterable(value, nodes_mask)
    #         elif key != 'edge_index':  # an edges att
    #             new_data_graph[key] = get_masked_iterable(value, edges_mask)
    #     new_data_graph['edge_index'] = edge_index_subgraph
    #     for key in data_graph.keys:
    #         assert key in new_data_graph.keys, key + 'not in new_data_graph'
    #     return new_data_graph

    @property
    def functions_vocab(self):
        return self._vocab.functions_vocab

    @property
    def all_function_names(self):
        return self._all_function_names

    @property
    def raw_dir(self):
        return osp.join(self.root, 'raw', self.stripped_or_original)

    @property
    def raw_file_names(self):
        raw_file_names = []
        for function_call_graphs in os.listdir(self.raw_dir):
            # Check that function features file is FUNCTION_CALL_GRAPHS_FILE_EXTENSION ('gexf')
            if os.path.splitext(function_call_graphs)[1] == PROGRAM_CALL_GRAPHS_FILE_EXTENSION:
                raw_file_names.append(function_call_graphs)
        return raw_file_names

    @property
    def processed_file_names(self):
        processed_file_names = []
        for function_call_graphs in os.listdir(self.raw_dir):
            # Check that function features file is FUNCTION_CALL_GRAPHS_FILE_EXTENSION ('gexf')
            splitted_file_name = os.path.splitext(function_call_graphs)
            if splitted_file_name[1] == PROGRAM_CALL_GRAPHS_FILE_EXTENSION:
                processed_file_names.append(splitted_file_name[0] + PROGRAMS_DATASET_FILE_EXTENSION)

        return processed_file_names

    @property
    def num_nodes(self):
        return self.data.edge_index.max().item() + 1

    def len(self):
        return self._len

    def get(self, idx):
        return self._examples[idx]

    def remove_from_graph_the_functions_not_in_dire(self):
        reduced_graphs_list = []
        for raw_path, processed_path in tqdm(zip(self.raw_paths, self.processed_paths)):
            try:
                graph = networkx.read_gexf(raw_path)
                raw_dire_program_data_path = os.path.splitext(raw_path)[0] + DIRE_PROGRAM_DATA_FILE_EXTENSION
                ##### Remove functions that don't appear in DIRE and sort functions to have same order in DIRE and graph. ####
                # TODO: Explore why dire output doesn't contain all methods graph contains
                mangled_function_names, demangled_function_names = get_mangled_and_demangled_names_from_networkx_graph(
                    graph)
                mangled_function_names_not_in_dire, demangled_function_names_not_in_dire = \
                    self.get_function_names_that_dont_exist_in_dire(raw_dire_program_data_path,
                                                                    mangled_function_names,
                                                                    demangled_function_names)

                for node_name, name in mangled_function_names_not_in_dire:
                    if node_name in graph.nodes:  # TODO: understand why some methods appear twice
                        graph.remove_node(node_name)

                reduced_graphs_list.append(graph)
            except:
                print('broken file')
                remove_broken_file(raw_path, processed_path)
        return reduced_graphs_list

    def process(self):
        self.reduced_graphs_list = self.remove_from_graph_the_functions_not_in_dire()
        self.initialize_vocab_if_needed(self.reduced_graphs_list)
        torch.save(self._vocab, self.vocab_dir)
        for raw_path, processed_path, reduced_graph in tqdm(zip(self.raw_paths, self.processed_paths,
                                                                self.reduced_graphs_list)):
            try:
                print("processing: " + raw_path)
                if self.only_small_dataset and len(reduced_graph) > self.small_program_size:
                    print('too large graph')
                    raise ('too large graph')
                self.insert_tokenized_demangled_names_into_graph(reduced_graph)
                self.insert_should_validate_on_per_function(reduced_graph)
                # get names again after the removal
                functions_data = self.get_appropriate_data_for_graph_examples(raw_path, reduced_graph)
                new_functions_data = []
                if self.load_nero_vectors:
                    nero_path = get_name_from_path(raw_path)
                    for function_data, node_info in zip(functions_data, reduced_graph.nodes.data()):
                        real_name = node_info[1]['mangled_function_names']
                        new_function_data = function_data
                        nero_id = get_nero_name_from_name(real_name) + '@' + nero_path
                        new_function_data.nero_id = nero_id
                        new_functions_data.append(new_function_data)

                        print(f'name -{nero_id} - in nero: {nero_id in nero_cache.examples.keys()}')
                        if nero_id in nero_cache.examples.keys():
                            global counter
                            counter += 1
                        else:
                            global counter_out
                            counter_out += 1
                            if nero_id[0] == '.':
                                global counter_point
                                counter_point += 1
                        print(f'counter in: {counter}')
                        print(f' not in: {counter_out}, of them start in point: {counter_point}')
                    functions_data = new_functions_data
                dire_datas = []
                graph_datas = []
                if self.use_subgraphs_method:
                    for i in range(self.number_of_subsamples):
                        new_data_graph, new_dire_data = self.get_random_subgraph_geometric_data(functions_data,
                                                                                                reduced_graph)
                        dire_datas.append(new_dire_data)
                        graph_datas.append(new_data_graph)
                else:
                    dire_data_tensor_dict = DireTensorDataset.functions_data_to_tensor_dict(functions_data,
                                                                                            self._vocab.variables_dire_vocab,
                                                                                            load_nero_vectors=self.load_nero_vectors)
                    new_dire_data = torch_geometric.data.Data.from_dict(dire_data_tensor_dict)
                    dire_datas.append(new_dire_data)
                    graph_datas.append(from_networkx(reduced_graph))
                num_examples_per_processed = len(dire_datas)
                torch.save((dire_datas, graph_datas, num_examples_per_processed), processed_path)
                print(f'file in {raw_path} is good')
            except Exception as e:
                print(e)
                print(f'file in {raw_path} is bad')
                remove_broken_file(raw_path, processed_path)

    def get_appropriate_data_for_graph_examples(self, raw_path, reduced_graph):
        mangled_function_names, demangled_function_names = \
            get_mangled_and_demangled_names_from_networkx_graph(reduced_graph)
        raw_dire_program_data_path = os.path.splitext(raw_path)[0] + DIRE_PROGRAM_DATA_FILE_EXTENSION
        functions_data = DireRawSingleProgramDataset.get_functions_data_from_json_path(
            raw_dire_program_data_path)
        indices_of_functions_data_sorted_by_graph_order = []
        for (node_name, mangled_func_name), (node_name, real_name) in zip(mangled_function_names,
                                                                          demangled_function_names):
            matching_functions_indices = [i for i, f in enumerate(functions_data) if
                                          f.mangled_name == mangled_func_name]
            assert len(matching_functions_indices) == 1
            # if not function_name_starts_with_sub(real_name):
            indices_of_functions_data_sorted_by_graph_order.append(matching_functions_indices[0])
        # remove all unreal names of functions from dire!
        if self.force_remove_function_names_from_dire:
            demangled_function_names.sort(key=lambda x: len(x[1]))
            for key, function_name in demangled_function_names:
                if not function_name_starts_with_sub(function_name):
                    inplace_change_word_occurences_in_file(raw_dire_program_data_path, function_name,
                                                           GENERIC_FUNCTION_NAME + str(key))
        functions_data = DireRawSingleProgramDataset.get_functions_data_from_json_path(
            raw_dire_program_data_path)
        functions_data = [functions_data[i] for i in indices_of_functions_data_sorted_by_graph_order]
        return functions_data

    def get_random_subgraph_geometric_data(self, functions_data, reduced_graph):
        number_of_nodes = len(functions_data)
        perm = torch.randperm(number_of_nodes)
        nodes_mask = perm[:self.size_of_subsamples]
        subgraph = reduced_graph.subgraph([str(i) for i in nodes_mask.numpy()])
        subgraph_data = from_networkx(subgraph)
        dire_data_tensor_dict = DireTensorDataset.functions_data_to_tensor_dict(
            get_masked_iterable(functions_data, nodes_mask),
            self._vocab.variables_dire_vocab)
        # new_data_graph = ProgramsDataset.data_graph_of_nodes_subset(graph_data, nodes_mask)
        new_data_graph = subgraph_data
        new_dire_data = torch_geometric.data.Data.from_dict(dire_data_tensor_dict)
        # if self.pre_filter is not None and not self.pre_filter(graph_data):
        #     continue
        if self.pre_transform is not None:
            new_data_graph = self.pre_transform(subgraph_data)
        return new_data_graph, new_dire_data

    # Ugly, but call this only when self.raw_file_names in initialized.
    def initialize_vocab_if_needed(self, reduced_graphs_list):
        if not hasattr(self, '_vocab') or self._vocab is None:
            print("Reading raw .jsonl files to build vocabulary")
            programs_data = []
            for raw_path in self.raw_paths:
                raw_dire_program_data_path = os.path.splitext(raw_path)[0] + DIRE_PROGRAM_DATA_FILE_EXTENSION
                programs_data.append(
                    DireRawSingleProgramDataset.get_functions_data_from_json_path(raw_dire_program_data_path))
            self._vocab = FullVocab(load_dire_from_files=False, all_programs_raw_function_data_list_list=programs_data,
                                    dire_vocab_file=DIRE_VOCABULARY_GENERATED_EXAMPLE_FILE,
                                    functions_vocab_file=FUNCTIONS_VOCABULARY_GENERATED_EXAMPLE_FILE)

        if self._vocab.functions_vocab is None:
            self.init_functions_names_list(reduced_graphs_list)
            self.init_functions_vocab()

    def init_functions_names_list(self, reduced_graphs_list):
        demangled_functions_names_list = []
        for reduced_graph in tqdm(reduced_graphs_list, unit='reduced graph'):
            mangled_function_names, demangled_function_names = \
                get_mangled_and_demangled_names_from_networkx_graph(reduced_graph)
            demangled_functions_names_list += demangled_function_names
        # demangled_functions_names_list = [demangled_name.lower() for demangled_name in demangled_functions_names_list]
        with open(os.path.join(os.path.split(self.vocab_dir)[0], 'functions_real_names_list.pkl'), 'wb+') as f:
            pickle.dump(demangled_functions_names_list, f)

    def init_functions_vocab(self):
        with open(os.path.join(os.path.split(self.vocab_dir)[0], 'functions_real_names_list.pkl'), 'rb') as f:
            demangled_functions_names_list = pickle.load(f)
        self._vocab.functions_vocab = FunctionsVocab(load_from_existing_file=False,
                                                     demangled_functions_names_list=demangled_functions_names_list,
                                                     vocab_file_prefix=self._vocab._functions_vocabulary_file_path,
                                                     vocabulary_size=self.functions_vocab_size)
        print(self._vocab.functions_vocab.get_id2word())

    def insert_tokenized_demangled_names_into_graph(self, graph):
        for node_info in graph.nodes.data():
            node_data = node_info[1]
            demangled_name = node_data[PROGRAM_GRAPH_FUNCTION_NODE_DEMANGLED_NAME]
            node_data[PROGRAM_GRAPH_FUNCTION_NODE_DEMANGLED_NAME_TOKENIZED] = \
                self._vocab.functions_vocab.encode_as_subtoken_ids(demangled_name)

    def insert_should_validate_on_per_function(self, graph):
        for node_info in graph.nodes.data():
            node_data = node_info[1]
            demangled_name = node_data[PROGRAM_GRAPH_FUNCTION_NODE_DEMANGLED_NAME]
            if (not demangled_name.startswith('sub')) or demangled_name.startswith('subX'):
                node_data[PROGRAM_GRAPH_FUNCTION_NODE_SHOULD_VALIDATE] = 1
            else:
                node_data[PROGRAM_GRAPH_FUNCTION_NODE_SHOULD_VALIDATE] = 0

    def get_function_names_that_dont_exist_in_dire(self, functions_json_path, mangled_function_names,
                                                   demangled_function_names):
        mangled_function_names_to_idx = dict()

        for i in range(len(mangled_function_names)):
            node_name, name = mangled_function_names[i]
            mangled_function_names_to_idx[name] = i

        demangled_function_names_to_idx = dict()

        for i in range(len(demangled_function_names)):
            node_name, name = demangled_function_names[i]
            demangled_function_names_to_idx[name] = i

        num_methods = len(mangled_function_names)

        asts = [None] * num_methods

        for json_function_line in open(functions_json_path, 'r').readlines():
            try:
                tree_json_dict = json.loads(json_function_line)
                function_ast = AbstractSyntaxTree.from_json_dict(tree_json_dict)
                if tree_json_dict['function'] in mangled_function_names_to_idx:
                    asts[mangled_function_names_to_idx[tree_json_dict['function']]] = function_ast
            except:
                print('problem with function name')
                raise ()

        bad_mangled_names = [(node_name, name) for (node_name, name) in \
                             mangled_function_names if \
                             asts[mangled_function_names_to_idx[name]] is None]

        bad_demangled_names = [(node_name, name) for (node_name, name) in \
                               demangled_function_names if \
                               asts[demangled_function_names_to_idx[name]] is None]

        return bad_mangled_names, bad_demangled_names


if __name__ == '__main__':
    delete_dataset_if_needed(reload_dataset=True, dataset_root_dir=nero_test_dataset_dirs.root_dir)
    MergeFunctionsDataIntoNetworkx(nero_test_dataset_dirs.decompiled_binaries_data_dir, nero_test_dataset_dirs.raw_dir)
    test_dataset = ProgramsDataset(nero_test_dataset_dirs.root_dir, load_dire_vocab_from_file=True,
                                   stripped_or_original=STRIPPED,
                                   only_small_dataset=True,
                                   small_program_size=1000,
                                   size_of_subsamples=100,
                                   number_of_subsamples=5
                                   )
