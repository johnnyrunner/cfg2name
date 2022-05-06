import argparse
import os
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path

from DatasetGeneration.decompiler_scripts.features_generator import FeatureGenerator
from config import DECOMPILED_BINARIES_DATA_DIR, STRIPPED_BINARIES_DIR, ORIGINAL_BINARIES_DIR, \
    ORIGINAL_FUNCTIONS_FEATURES_FOLDER_NAME
from utils.function_utils import from_debug_name_to_name


def check_if_binary_already_preprocessed(lib_name):
    path_to_function_features = os.path.join(DECOMPILED_BINARIES_DATA_DIR, ORIGINAL_FUNCTIONS_FEATURES_FOLDER_NAME)
    Path(path_to_function_features).mkdir(exist_ok=True, parents=True)
    binaries_preprocessed_outputs = os.listdir(path_to_function_features)
    binaries_preprocessed_names = [binary_preprocessed_output.split('.stripped')[0] for binary_preprocessed_output
                                   in binaries_preprocessed_outputs]
    return lib_name in binaries_preprocessed_names


def preprocess_all_binaries_in_folder(original_binaries_dir: str, output_dir: str, stripped_binaries_dir: str = None,
                                      remove_dynamic_symbols_table: bool = False, relevant_libs_list_path: str = None,
                                      num_parallel_threads: int = 8, use_stripped_path: bool = True):
    features_generator = FeatureGenerator(output_dir, stripped_binaries_dir, use_stripped_path)
    if relevant_libs_list_path is not None:
        relevant_libs_list = [from_debug_name_to_name(name) if from_debug_name_to_name(name) is not None
                              else '' for name in open(relevant_libs_list_path, 'r').readlines()]
    else:
        relevant_libs_list = None

    executor = ThreadPoolExecutor(max_workers=num_parallel_threads)

    for root, dirs, files in os.walk(original_binaries_dir):
        if len(files) > 0:
            executor.submit(process_all_binaries_in_dir,
                            features_generator, files, relevant_libs_list, remove_dynamic_symbols_table,
                            root)
            print('sent thread')


def process_all_binaries_in_dir(features_generator, files, relevant_libs_list, remove_dynamic_symbols_table, root):
    for original_binary in files:
        if check_if_binary_already_preprocessed(original_binary):
            print(f'Binary already processed : {original_binary}')
            continue
        if relevant_libs_list is not None and os.path.split(root)[-1] not in relevant_libs_list:
            print(f'File not in list : {original_binary}')
            continue
        original_binary_dir = os.path.join(root, original_binary)
        features_generator.preprocess_binary_in_certain_location(original_binary_dir,
                                                                 remove_dynamic_symbols_table)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the decompiler to generate a corpus.")
    parser.add_argument('--original_binaries_dir',
                        metavar='ORIGINAL_BINARIES_DIR',
                        help="directory containing binaries with symbols",
                        default=ORIGINAL_BINARIES_DIR
                        )

    parser.add_argument('--stripped_binaries_dir',
                        metavar='STRIPPED_BINARIES_DIR',
                        help="directory containing stripped binaries",
                        default=STRIPPED_BINARIES_DIR
                        )

    parser.add_argument('--output_dir',
                        metavar='OUTPUT_DIR',
                        help="output directory",
                        default=DECOMPILED_BINARIES_DATA_DIR
                        )

    parser.add_argument('--remove_dynamic_symbols_table',
                        metavar='remove_dynamic_symbols_table',
                        help="remove_dynamic_symbols_table",
                        default=False
                        )

    parser.add_argument('--relevant_libs_list',
                        metavar='relevant_libs_list',
                        help="a filepath of the list of wanted files to take care of",
                        default=False
                        )
    args = parser.parse_args()
    # preprocess_all_binaries_in_folder(args.original_binaries_dir, args.output_dir, args.stripped_binaries_dir)
    preprocess_all_binaries_in_folder(args.original_binaries_dir, args.output_dir)
    # preprocess_all_binaries_in_folder(args.original_binaries_dir, args.output_dir,
    #                                   relevant_libs_list_path=apt_lib_names_file_path.format('a'))
    # preprocess_all_binaries_in_folder(BINARIES_DIR, args.output_dir)
