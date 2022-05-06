import os
import pathlib
import pickle
from typing import List

import networkx
import pandas
from collections import Counter

import matplotlib.pyplot as plt
from tqdm import tqdm

from Datasets.graph_features_builder import get_mangled_and_demangled_names_from_networkx_graph
from config import PROGRAM_GRAPH_DATASET_ROOT_DIR, DECOMPILED_BINARIES_DATA_DIR, \
    PROGRAM_GRAPH_DATASET_RAW_DIR, FUNCTIONS_VOCABULARY_EXAMPLE_DIR, DATA_EXPLORATION_CACHE
from utils.function_utils import flatten_python_list
from utils.graph_utils import MergeFunctionsDataIntoNetworkx
from vocabulary_builder.functions_vocab import FunctionsVocab

import numpy as np


# def merge_functions_data_into_networkx():
#     for file_path in pathlib.Path(PROGRAM_GRAPH_DATASET_ROOT_DIR).glob(f"**/*"):
#         if os.path.isfile(file_path):
#             os.remove(file_path)
#     print("Merging data from function call graph into dire AST data, and moving files into raw folder...")
#     MergeFunctionsDataIntoNetworkx(DECOMPILED_BINARIES_DATA_DIR, PROGRAM_GRAPH_DATASET_RAW_DIR)


def load_functions_names_list_from_files(
        # gexf_files_dir='D:\\routinio_data\\data_new\\decompiled_binaries\\function_call_graphs',
        gexf_files_dir='/media/jonathan/New Volume/routinio_data/data_new_dire/decompiled_binaries_test/function_call_graphs',
        build: bool = False):
    functions_names_list = []
    all_demangled_functions_names_list = []
    print('load_functions_names_list_from_files')
    for gexf_file_name in tqdm(os.listdir(gexf_files_dir)):
        if gexf_file_name.split('.')[-1] == 'gexf' and '.stripped' not in gexf_file_name:
            original_graph = networkx.read_gexf(os.path.join(gexf_files_dir, gexf_file_name))
            _, demangled_functions_names_list = get_mangled_and_demangled_names_from_networkx_graph(
                original_graph)
            functions_names_list += [name for index, name in demangled_functions_names_list]
            all_demangled_functions_names_list += demangled_functions_names_list
    return functions_names_list, all_demangled_functions_names_list


def strings_list_histogram(strings_list, to_plot: bool = False, title='no title given', percentile=95):
    letter_counts = Counter(strings_list)
    df = pandas.DataFrame.from_dict(letter_counts, orient='index')
    print(df)
    if to_plot:
        # df.plot(kind='bar')
        # plt.show()
        plt.hist(df[0], bins=60, range=(1, np.percentile(df[0], percentile)))
        plt.title(f'{title} histogram - top {percentile} percentile of words')
        plt.xlabel('word occurences')
        plt.ylabel('number of words occured this much')
        plt.show()

    number_of_functions = len(strings_list)
    print(f'number of words - {number_of_functions}')
    number_of_functions_w_sum_in_name = sum(["sub" in name for name in strings_list])
    print(f'number of words with "sub" in name - {number_of_functions_w_sum_in_name}')
    print(
        f'percentage of words with sub in name {100 * number_of_functions_w_sum_in_name / number_of_functions} %')


def explore_vocab(functions_vocab: FunctionsVocab, all_sentences: List[str]):
    print('vocab size')
    print(functions_vocab._vocab_entry.vocab_size)
    print(len(functions_vocab._vocab_entry.word2id))
    print(functions_vocab._vocab_entry.word2id)
    print(len(all_sentences))
    print(all_sentences)
    functions_subtokens_ids = []
    functions_subtokens = []
    for function_name in tqdm(all_sentences, unit='function name'):
        function_subtoken_ids = functions_vocab.normal_encode_as_subtoken_ids(function_name)
        function_subtokens = functions_vocab.encode_as_subtokens_list(function_name)
        functions_subtokens_ids.append(function_subtoken_ids)
        functions_subtokens.append(function_subtokens)
        print(function_name)
        print(function_subtokens)
    strings_list_histogram(flatten_python_list(functions_subtokens), to_plot=True, title='tokens')
    strings_list_histogram(all_sentences, to_plot=True, title='full names')


def get_function_vocab(use_cache: bool =False):
    cache_base_name = 'functions_vocab_1'
    cache_file_path = os.path.join(DATA_EXPLORATION_CACHE, cache_base_name)
    if not use_cache or not os.path.exists(cache_file_path):
        functions_names_list, demangled_names_list = load_functions_names_list_from_files(build=True)
        functions_vocab = FunctionsVocab(False, demangled_functions_names_list=demangled_names_list,
                                         vocab_file_prefix=os.path.join(FUNCTIONS_VOCABULARY_EXAMPLE_DIR,
                                                                        'vocab_exploration_example'),
                                         vocabulary_size=2000,
                                         character_coverage=0.999)
        if use_cache:
            with open(cache_file_path, 'wb+') as f:
                pickle.dump((functions_names_list, functions_vocab), f)
                print('dumped')
                print(f'to: {cache_file_path}')
    else:
        with open(cache_file_path, 'rb') as f:
            functions_names_list, functions_vocab = pickle.load(f)
            print('loaded all_subtokens_concatenated_df')
            print(f'from: {cache_file_path}')

    return functions_names_list, functions_vocab


if __name__ == '__main__':
    # merge_functions_data_into_networkx()
    functions_names_list, functions_vocab = get_function_vocab()
    explore_vocab(functions_vocab, functions_names_list)
