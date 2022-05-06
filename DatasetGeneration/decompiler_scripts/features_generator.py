import argparse
import logging
import os
import pickle
import subprocess
import time
from pathlib import Path

import networkx

from config import IDA_PYTHON_HOME, IDA_PYTHON_LIB, DECOMPILATION_TIMEOUT, IDA64_PATH, FUNCTION_CALL_GRAPHS, \
    DECOMPILED_BINARIES_DATA_DIR, LOCAL_TEMP_DIR, PROGRAM_CALL_GRAPHS_FILE_EXTENSION, MIN_SIZE_OF_LEGIT_SOFTWARE, \
    LARGEST_SIZE_RELEVANT_IN_BYTES, SMALLEST_SIZE_RELEVANT_IN_BYTES, LOCAL_TEMP_DIR, PROGRAM_CALL_GRAPHS_FILE_EXTENSION, MIN_SIZE_OF_LEGIT_SOFTWARE, \
    LARGEST_SIZE_RELEVANT_IN_BYTES, SMALLEST_SIZE_RELEVANT_IN_BYTES, ORIGINAL_FUNCTIONS_FEATURES_FOLDER_NAME, \
    STRIPPED_FUNCTIONS_FEATURES_FOLDER_NAME, ORIGINAL_BINARIES_DIR, ORIGINAL_BINRARIES_DIRS, SLEEP_TIME
from utils.files_utils import switch_dir_to_ubuntu
from utils.graph_utils import print_and_save_graph, demangle_function_names_in_graph_labels, add_edges_to_graph
from utils.function_utils import timeit
import tempfile
from utils import files_utils


class FeatureGenerator:
    def __init__(self, output_dir, stripped_binaries_dir=None, use_stripped_path: bool = True,
                 decompilation_timeout=DECOMPILATION_TIMEOUT, ida_version=IDA64_PATH):
        self.env = self.gen_env(output_dir)
        self.stripped_binaries_dir = stripped_binaries_dir
        self.use_stripped_path = use_stripped_path
        tempfile.tempdir = LOCAL_TEMP_DIR

        statyre_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)))
        self.collect_vars_script_dir = os.path.join(statyre_dir, 'collect.py')
        self.original_dire_script_dir = os.path.join(statyre_dir, 'dump_trees.py')
        self.call_graph_script_dir = os.path.join(statyre_dir, 'functions_call_graph.py')
        self.decompilation_timeout = decompilation_timeout

        self.ida_version = ida_version

    def gen_env(self, output_dir):
        files_utils.make_directory(output_dir)
        env = os.environ.copy()
        env['PYTHONPATH'] = IDA_PYTHON_LIB
        env['PYTHONHOME'] = IDA_PYTHON_HOME

        # Check for/create output directories
        self.output_dir = os.path.abspath(output_dir)
        env['OUTPUT_DIR'] = self.output_dir
        return env

    def run_decompiler(self, file_path, script):
        """Run a decompiler script.

        Keyword arguments:
        file_path -- the binary to be decompiled
        env -- an os.environ mapping, useful for passing arguments
        script -- the script file to run
        timeout -- timeout in seconds (default no timeout)
        """
        try:
            file_copy_name = files_utils.create_temp_copy(file_path)
            idacall = [self.ida_version, '-B', f'-S{script}', file_copy_name]
            if script == self.call_graph_script_dir:
                output = subprocess.check_output(idacall, env=self.env, timeout=self.decompilation_timeout, shell=True)
            else:
                output = subprocess.check_output(idacall, env=self.env, timeout=self.decompilation_timeout)
            return output
        except subprocess.TimeoutExpired:
            print("Timed out\n")
            return False

    def configure_for_ida_call(self, binary, functions_features_folder_name: str = ORIGINAL_FUNCTIONS_FEATURES_FOLDER_NAME):
        self.env['BINARY_NAME_PREFIX'] = binary
        self.env['FUNCTIONS_FEATURES_FOLDER_NAME'] = functions_features_folder_name
        self.env['IDALOG'] = os.path.join(self.output_dir, 'logs', f"ida_{binary}.log")
        Path(os.path.join(self.output_dir, 'logs')).mkdir(exist_ok=True)

    def preprocess_binary_in_certain_location(self, original_binary_dir,
                                              remove_dynamic_symbols_table: bool = False):
        lib_size = Path(original_binary_dir).stat().st_size
        if lib_size > LARGEST_SIZE_RELEVANT_IN_BYTES:
            print(f'Lib is too large, shouldn\'t try: size is {lib_size}, max size is {LARGEST_SIZE_RELEVANT_IN_BYTES}')
            return
        if lib_size < SMALLEST_SIZE_RELEVANT_IN_BYTES:
            print(
                f'Lib is too small, shouldn\'t try: size is {lib_size}, min size is {SMALLEST_SIZE_RELEVANT_IN_BYTES}')
            return
        if self.stripped_binaries_dir is not None:
            # this line is based on the only difference between original_path and stripped_path - the word 'out'
            stripped_binary_file_dir = original_binary_dir.replace('with', 'without')
            if not os.path.exists(stripped_binary_file_dir):
                return
        else:
            stripped_binary_file_dir = None

        print(f"File {os.path.split(original_binary_dir)[-1]}")
        if not self.extract_features_for_binary(original_binary_dir, stripped_binary_file_dir,
                                                remove_dynamic_symbols_table=remove_dynamic_symbols_table):
            print(f'didnt succeed on process of {os.path.split(original_binary_dir)[-1]}')

    @timeit
    def extract_features_for_binary(self, original_binary_dir: str, stripped_binary_dir: str = None,
                                    starts_with_dbg_symbols: bool = True, remove_dynamic_symbols_table: bool = False):
        '''

        :param original_binary_dir: the binary file it works on name
        :return:
        '''
        print('started extractinjg features for binary')
        # try:
        if starts_with_dbg_symbols:
            print('yes stripped dir!')
            ##### configure ####
            original_path, original_binary_name = os.path.split(original_binary_dir)
            self.configure_for_ida_call(original_binary_name)
            # ##### collect var names ####
            if not self.collect_vars(original_binary_dir):
                print('Failed to collect - IDA is not ok with it')
                return False
            print(f"{original_binary_name} collected vars")

            #### collect functions names ####
            self.extract_call_graph(original_binary_name, original_binary_dir)
            print(f"{original_binary_name} extracted call graph")

            ### print jsonl of non-stripped
            self.run_decompiler(original_binary_dir, self.original_dire_script_dir)
            print(f"{original_binary_dir} dumpped trees on original")

            ### copy the file into a place we'll find later on
            originals_path = os.path.join(self.output_dir, 'original_binaries')
            Path(originals_path).mkdir(exist_ok=True)
            new_original_binary_dir = os.path.join(originals_path, original_binary_name)
            subprocess.call(
                ['ubuntu', 'run', 'cp', switch_dir_to_ubuntu(original_binary_dir),
                 switch_dir_to_ubuntu(new_original_binary_dir)])
            time.sleep(SLEEP_TIME)

            ### save the original binary path to a small file
            originals_dirs_path = os.path.join(self.output_dir, ORIGINAL_BINRARIES_DIRS)
            Path(originals_dirs_path).mkdir(exist_ok=True)
            new_original_binary_dir_dir = os.path.join(originals_dirs_path, original_binary_name + '.txt')
            with open(new_original_binary_dir_dir, 'w+') as f:
                f.write(original_binary_dir)


            ##### extract the asts and the code of the functions ####
            if stripped_binary_dir is not None:
                stripped_path, stripped_binary_name = os.path.split(stripped_binary_dir)
            else:
                stripped_path = os.path.join(self.output_dir, 'stripped_binaries')
                Path(stripped_path).mkdir(exist_ok=True)
                stripped_binary_name = original_binary_name + '.stripped'
                stripped_binary_dir = os.path.join(stripped_path, stripped_binary_name)
                subprocess.call(
                    ['ubuntu', 'run', 'cp', switch_dir_to_ubuntu(original_binary_dir),
                     switch_dir_to_ubuntu(stripped_binary_dir)])
                time.sleep(2)
                subprocess.call(
                    ['ubuntu', 'run', 'strip', '--strip-all',
                     switch_dir_to_ubuntu(stripped_binary_dir)])  # was strip-debug
                time.sleep(2)
                if remove_dynamic_symbols_table:
                    # TODO: change to programatic stripping of the output
                    subprocess.call(
                        ['ubuntu', 'run', 'strip', '-R .dynsym',
                         switch_dir_to_ubuntu(stripped_binary_dir)])

                print(f"{original_binary_dir} stripped")
            # Dump the trees.
            # No timeout here, we know it'll run in a reasonable amount of
            # time and don't want mismatched files
            self.configure_for_ida_call(stripped_binary_name, STRIPPED_FUNCTIONS_FEATURES_FOLDER_NAME)

            self.run_decompiler(stripped_binary_dir, self.original_dire_script_dir)
            print(f"{stripped_binary_dir} dumpped trees on stripped")

            ##### extract the functions call graph ####
            self.extract_call_graph(stripped_binary_name, stripped_binary_dir)
            print(f"{stripped_binary_name} extracted call graph on stripped")

        else:  # hence, the file starts without debug symbols
            print('no stripped dir!')
            file_path = self.configure_for_ida_call(original_binary_dir)
            if not self.run_decompiler(file_path, self.original_dire_script_dir):
                return False

            ##### extract the functions call graph ####
            self.extract_call_graph(original_binary_dir, file_path)

            ########## add the debug symbols here #####
            ######################################3###

            ##### collect var names ####
            if not self.collect_vars(file_path):
                return False

            ##### collect functions names ####
            self.extract_call_graph(original_binary_dir, file_path)
        # except:
        #     print('Timeout, probably')
        #     return False
        print(f"{stripped_binary_name} finished all processing!")

        return True

    def extract_call_graph(self, binary_name, file_path):
        call_graphs_path = os.path.join(self.output_dir, FUNCTION_CALL_GRAPHS)
        Path(call_graphs_path).mkdir(parents=True, exist_ok=True)
        self.env['GDL_FILE_DIR'] = os.path.join(call_graphs_path, binary_name) + '.gdl'
        self.env['GML_FILE_DIR'] = os.path.join(call_graphs_path, binary_name) + '.gml'
        self.env['CHILDREN_PICKLE_DIR'] = os.path.join(call_graphs_path, binary_name) + '.pkl'

        self.run_decompiler(file_path, self.call_graph_script_dir)

        # names from within the DUMP_CALL_GRAPH - cannot be shared because ida won't support multiple files #TODO: check if ida won't support
        graph_file_path_ubuntu = files_utils.switch_dir_to_ubuntu(self.env['GDL_FILE_DIR'])
        gml_file_path_ubuntu = files_utils.switch_dir_to_ubuntu(self.env['GML_FILE_DIR'])

        for path in files_utils.execute(['ubuntu', 'run',
                                         'graph-easy', graph_file_path_ubuntu,
                                         '--as=graphml --output=' + gml_file_path_ubuntu],
                                        self.output_dir):
            print(path)
        logging.info('gml file has been saved to {}\n'.format(self.env['GML_FILE_DIR']))
        graph = networkx.read_graphml(self.env['GML_FILE_DIR'])
        graph = demangle_function_names_in_graph_labels(graph)

        all_black_dict = {k: 'black' for k in graph.edges}
        networkx.set_edge_attributes(graph, all_black_dict, name='color')
        graph = add_edges_to_graph(graph, self.env['CHILDREN_PICKLE_DIR'])

        print_and_save_graph(graph, os.path.join(self.output_dir, FUNCTION_CALL_GRAPHS, binary_name), is_save=True)

        gexf_file_path = os.path.join(self.output_dir, FUNCTION_CALL_GRAPHS,
                                      binary_name) + PROGRAM_CALL_GRAPHS_FILE_EXTENSION
        networkx.write_gexf(graph, gexf_file_path)  # to read with networkx.read_gexf(gexf_file_path)
        return gexf_file_path

    def collect_functions_names(self, file_path):
        print(f"Collecting function names from {file_path}")
        collected_functions_names = tempfile.NamedTemporaryFile(delete=False)
        collected_function_name = collected_functions_names.name
        collected_functions_names.close()

        # First collect variables
        self.env['COLLECTED_FUNCTIONS'] = collected_function_name
        file_copy_name = files_utils.create_temp_copy(file_path)

        # Timeout after 3000 seconds for first run
        self.run_decompiler(file_copy_name, self.collect_functions_script_dir)

        try:
            print(collected_function_name)
            with open(collected_function_name, 'rb') as collected_functions_names:
                collected_functions_data = collected_functions_names.read().replace(b'\r\n', b'\n')
                print(collected_functions_data)
            if not pickle.loads(collected_functions_data):
                print("No functions collected\n")
                return False
        except:
            print("No functions collected\n")
            return False

            return True

    def collect_vars(self, file_path):
        print(f"Collecting var names from {file_path}")
        collected_vars = tempfile.NamedTemporaryFile(delete=False)
        collected_vars_file_name = collected_vars.name
        collected_vars.close()

        # First collect variables
        self.env['COLLECTED_VARS'] = collected_vars_file_name
        file_copy_name = files_utils.create_temp_copy(file_path)

        # Timeout after 3000 seconds for first run
        self.run_decompiler(file_copy_name, self.collect_vars_script_dir)

        try:
            print(collected_vars_file_name)
            with open(collected_vars_file_name, 'rb') as collected_vars:
                collected_vars_data = collected_vars.read().replace(b'\r\n', b'\n')
                print(collected_vars_data)
            if not pickle.loads(collected_vars_data) or len(collected_vars_data) < MIN_SIZE_OF_LEGIT_SOFTWARE:
                print("No variables collected\n")
                return False
        except:
            print("No variables collected\n")
            return False

        return True
