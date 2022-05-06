import os

import networkx
from tqdm import tqdm

from Datasets.graph_features_builder import get_mangled_and_demangled_names_from_networkx_graph

import matplotlib.pyplot as plt


def load_gexf_functions_numbers(
        gexf_files_dir='D:\\routinio_data\\new_data\\decompiled_binaries\\function_call_graphs',
        build: bool = False):
    original_graphs_functions_number = []
    stripped_graphs_functions_number = []
    count =0
    for gexf_file_name in tqdm(os.listdir(gexf_files_dir)):
        if gexf_file_name.split('.')[-1] == 'gexf' and '.stripped' not in gexf_file_name:
            original_graph = networkx.read_gexf(os.path.join(gexf_files_dir, gexf_file_name))
            stripped_name = '.'.join(gexf_file_name.split('.')[:-1]) + '.stripped.' + gexf_file_name.split('.')[-1]
            try:
                stripped_graph = networkx.read_gexf(os.path.join(gexf_files_dir, stripped_name))
            except:
                # print('didnt found stripped graph')
                count += 1
                continue
            _, demangled_functions_names_list = get_mangled_and_demangled_names_from_networkx_graph(
                original_graph)
            _, demangled_stripped_functions_names_list = get_mangled_and_demangled_names_from_networkx_graph(
                stripped_graph)
            original_graphs_functions_number.append(len(demangled_functions_names_list))
            stripped_graphs_functions_number.append(len(demangled_stripped_functions_names_list))
    print(f'in total, did not found {count} graphs')
    return original_graphs_functions_number, stripped_graphs_functions_number


def plot_original_vs_stripped_number_of_functions():
    original_graphs_functions_number, stripped_graphs_functions_number = load_gexf_functions_numbers()
    plt.plot(original_graphs_functions_number, stripped_graphs_functions_number, '.')
    plt.xlabel('original number of functions')
    plt.ylabel('stripped number of functions')
    plt.title('number of functions - original vs. stripped graphs')
    plt.show()


if __name__ == '__main__':
    plot_original_vs_stripped_number_of_functions()
