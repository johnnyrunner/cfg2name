import logging
from datetime import date
from pathlib import Path
import os
# decompilation
import torch

logger = logging.Logger('logger')
logger.propagate = False

IDA64_PATH = "C:/Program Files/IDA 7.0/ida64.exe"
IDA32_PATH = "C:/Program Files/IDA 7.0/ida.exe"
IDA_PYTHON_HOME = "C:/Python27"
IDA_PYTHON_LIB = "C:/Python27/Lib"

IDA64_PATH = "C:\\Program Files\\IDA 7.0\\ida64.exe"
IDA32_PATH = "C:\\Program Files\\IDA 7.0\\ida.exe"
IDA_PYTHON_HOME = "C:\\Python27"
IDA_PYTHON_LIB = "C:\\Python27\\Lib"
DECOMPILATION_TIMEOUT = 30 * 60
# LARGEST_SIZE_RELEVANT_IN_BYTES = 1_000_000
# SMALLEST_SIZE_RELEVANT_IN_BYTES = 100_000
LARGEST_SIZE_RELEVANT_IN_BYTES = 20_000
SMALLEST_SIZE_RELEVANT_IN_BYTES = 1_000

ROUTINIO_DIR = Path(__file__).parent.absolute()

DRIVE_D_LOCATION = '/home/jonathan/Desktop/Thesis/routinio2.1'  # os.environ["ROUTINIO_DATA_DIR"]
# ALL_OUTPUTS_DIR = os.path.join(DRIVE_D_LOCATION, 'exploration_functions') # 155k functions
ALL_DATASETS_DIR = os.path.join(DRIVE_D_LOCATION, 'data_dire_trial')

Path(DRIVE_D_LOCATION).mkdir(exist_ok=True, parents=True)
Path(ALL_DATASETS_DIR).mkdir(exist_ok=True, parents=True)

DECOMPILED_BINARIES_DATA_DIR = os.path.join(ALL_DATASETS_DIR, 'decompiled_binaries')
DECOMPILED_BINARIES_TEST_DIR = os.path.join(ALL_DATASETS_DIR, 'decompiled_binaries_test')
ORIGINAL_BINARIES_DIR = os.path.join(ALL_DATASETS_DIR, 'binaries_with_symbols')
STRIPPED_BINARIES_DIR = os.path.join(ALL_DATASETS_DIR, 'binaries_without_symbols')
BINARIES_DIR = os.path.join(ALL_DATASETS_DIR, 'binaries')

Path(DECOMPILED_BINARIES_DATA_DIR).mkdir(exist_ok=True)
Path(DECOMPILED_BINARIES_TEST_DIR).mkdir(exist_ok=True)
Path(ORIGINAL_BINARIES_DIR).mkdir(exist_ok=True)
Path(STRIPPED_BINARIES_DIR).mkdir(exist_ok=True)


def make_dataset_dir(dataset_root_dir):
    dataset_raw_dir = os.path.join(dataset_root_dir, 'raw')
    processed_dir = os.path.join(dataset_root_dir, 'processed')
    Path(dataset_root_dir).mkdir(exist_ok=True)
    Path(dataset_raw_dir).mkdir(exist_ok=True)
    Path(processed_dir).mkdir(exist_ok=True)
    return dataset_raw_dir, processed_dir

class DatasetDirs:
    def __init__(self, root_dir,
                 raw_dir,
                 decompiled_binaries_data_dir=DECOMPILED_BINARIES_DATA_DIR):
        self.root_dir = root_dir
        self.raw_dir = raw_dir
        self.decompiled_binaries_data_dir = decompiled_binaries_data_dir

    @staticmethod
    def make_dataset_dir_for_name_and_location(name, decompiled_binaries_data_dir, all_datasets_dir=ALL_DATASETS_DIR):
        new_dataset_root_dir = os.path.join(all_datasets_dir, name)
        new_dataset_raw_dir, _ = make_dataset_dir(new_dataset_root_dir)
        return DatasetDirs(
            root_dir=new_dataset_root_dir,
            raw_dir=new_dataset_raw_dir,
            decompiled_binaries_data_dir=decompiled_binaries_data_dir
        )


NERO_TRAIN_ORIGIN_PROGRAM_GRAPH_DATASET_ROOT_DIR = os.path.join(ALL_DATASETS_DIR, 'nero_binaries_train')
NERO_VALIDATION_ORIGIN_PROGRAM_GRAPH_DATASET_ROOT_DIR = os.path.join(ALL_DATASETS_DIR, 'nero_binaries_validate')
NERO_TEST_ORIGIN_PROGRAM_GRAPH_DATASET_ROOT_DIR = os.path.join(ALL_DATASETS_DIR, 'nero_binaries_test')

nero_train_dataset_dirs = DatasetDirs.make_dataset_dir_for_name_and_location('nero_binaries_train_dataset', NERO_TRAIN_ORIGIN_PROGRAM_GRAPH_DATASET_ROOT_DIR)
nero_validation_dataset_dirs = DatasetDirs.make_dataset_dir_for_name_and_location('nero_binaries_validation_dataset', NERO_VALIDATION_ORIGIN_PROGRAM_GRAPH_DATASET_ROOT_DIR)
nero_test_dataset_dirs = DatasetDirs.make_dataset_dir_for_name_and_location('nero_binaries_test_dataset', NERO_TEST_ORIGIN_PROGRAM_GRAPH_DATASET_ROOT_DIR)
nero_small_train_dataset_dirs = DatasetDirs.make_dataset_dir_for_name_and_location('nero_small_binaries_train_dataset', NERO_TRAIN_ORIGIN_PROGRAM_GRAPH_DATASET_ROOT_DIR)

MAX_FILES_NUMBER = 4500

test_edges_origin_files = os.path.join(ALL_DATASETS_DIR, 'test_edges_new')
test_edges_dataset_dirs = DatasetDirs.make_dataset_dir_for_name_and_location('test_edges_dataset_dirs_dataset', test_edges_origin_files)


only_graphs_origin_files = os.path.join(ALL_DATASETS_DIR, 'only_graphs')
only_graphs_dataset_dirs = DatasetDirs.make_dataset_dir_for_name_and_location('only_graphs_dataset', only_graphs_origin_files)

all_programs_dataset_files = '/media/jonathan/New Volume/routinio_data/data_new_dire/decompiled_binaries_test'
datasets_location = '/media/jonathan/New Volume/routinio_data/datasets'

all_programs_dataset_dirs = DatasetDirs.make_dataset_dir_for_name_and_location('all_programs_dataset', all_programs_dataset_files, datasets_location)

all_programs_dataset_files_validation = '/media/jonathan/New Volume/routinio_data/data_new_dire/decompiled_binaries_validation_data'
all_programs_dataset_dirs_validation = DatasetDirs.make_dataset_dir_for_name_and_location('all_programs_dataset_validation', all_programs_dataset_files_validation, datasets_location)

#should be depracated, but not important
PROGRAM_GRAPH_DATASET_ROOT_DIR = os.path.join(ALL_DATASETS_DIR, 'final_stage_dataset')
PROGRAM_GRAPH_DATASET_RAW_DIR, _ = make_dataset_dir(PROGRAM_GRAPH_DATASET_ROOT_DIR)
SMALL_PROGRAM_GRAPH_DATASET_ROOT_DIR = os.path.join(ALL_DATASETS_DIR, 'small_final_stage_dataset')
make_dataset_dir(SMALL_PROGRAM_GRAPH_DATASET_ROOT_DIR)
#should be depracated, but not important


LOCAL_TEMP_DIR = os.path.join(ALL_DATASETS_DIR, 'tmp')
Path(LOCAL_TEMP_DIR).mkdir(exist_ok=True)

ORIGINAL_BINRARIES_DIRS = 'original_binaries_dirs'

FUNCTION_CALL_GRAPHS = 'function_call_graphs'
NUM_PARAM_TYPE = 'num'  # TODO: enuym
ORIGINAL_FUNCTIONS_FEATURES_FOLDER_NAME = 'functions_features_original'
STRIPPED_FUNCTIONS_FEATURES_FOLDER_NAME = 'functions_features_stripped'

FUNCTION_CALL_GRAPHS_DIRECTORY = 'function_call_graphs'
DIRE_PROGRAM_DATA_FILE_EXTENSION = '.jsonl'
PROGRAM_CALL_GRAPHS_FILE_EXTENSION = '.gexf'
PROGRAMS_DATASET_FILE_EXTENSION = '.pt'

YES = 'YES'

DIRE_AST_GRAPH_CONNECTIONS = ['top_down', 'bottom_up', 'terminals', 'variable_master_nodes', 'func_root_to_arg']

AST_DIRE_ENDING = '.astpt'

JSON_AST_NODE_ATTRIBUTE = 'JSON_AST'
AST_NODE_ATTRIBUTE = 'AST'
RAW_CODE_ATTRIBUTE = 'raw_code'

NUM_ARGUMENTS = 2
ARGUMENT_IDX_DIRE_CONFIG_FILE = 1

VOCABULARY_EXAMPLE_DIR = os.path.join(ALL_DATASETS_DIR, "vocabularies/vocab_example")
DIRE_VOCABULARY_EXAMPLE_DIR = os.path.join(VOCABULARY_EXAMPLE_DIR, "dire_vocab/")
FUNCTIONS_VOCABULARY_EXAMPLE_DIR = os.path.join(VOCABULARY_EXAMPLE_DIR, "functions_vocab/")

Path(VOCABULARY_EXAMPLE_DIR).mkdir(exist_ok=True, parents=True)
Path(DIRE_VOCABULARY_EXAMPLE_DIR).mkdir(exist_ok=True, parents=True)
Path(FUNCTIONS_VOCABULARY_EXAMPLE_DIR).mkdir(exist_ok=True, parents=True)

VOCABULARY_GENERATED_EXAMPLE_DIR = os.path.join(ALL_DATASETS_DIR, "vocabularies/vocab_generated_example")
EXAMPLE_VARS_VOCAB_GENERATED = os.path.join(VOCABULARY_GENERATED_EXAMPLE_DIR, "dire_vocab/")
EXAMPLE_FUNCTIONS_VOCAB_GENERATED = os.path.join(VOCABULARY_GENERATED_EXAMPLE_DIR, "functions_vocab/")

Path(VOCABULARY_GENERATED_EXAMPLE_DIR).mkdir(exist_ok=True, parents=True)
Path(EXAMPLE_VARS_VOCAB_GENERATED).mkdir(exist_ok=True, parents=True)
Path(EXAMPLE_FUNCTIONS_VOCAB_GENERATED).mkdir(exist_ok=True, parents=True)

DIRE_VOCABULARY_EXAMPLE_FILE = os.path.join(DIRE_VOCABULARY_EXAMPLE_DIR, "vocab")
DIRE_VOCABULARY_EXAMPLE_FILE = '/home/jonathan/Desktop/Thesis/routinio2.1/dire_neural_model/data/vocab_hybrid_pretrained/vocab'

FUNCTIONS_VOCABULARY_EXAMPLE_FILE = os.path.join(FUNCTIONS_VOCABULARY_EXAMPLE_DIR, "vocab")

DIRE_VOCABULARY_GENERATED_EXAMPLE_FILE = os.path.join(EXAMPLE_VARS_VOCAB_GENERATED, "vocab")
FUNCTIONS_VOCABULARY_GENERATED_EXAMPLE_FILE = os.path.join(EXAMPLE_FUNCTIONS_VOCAB_GENERATED, "vocab")

DIRE_DEFAULT_CONFIG_GNN_DIR = os.path.join(ROUTINIO_DIR, "dire_neural_model/data/config/config.gnn.jsonnet")
DIRE_DEFAULT_CONFIG_HYBRID_DIR = os.path.join(ROUTINIO_DIR,
                                              "dire_neural_model/data/config/hybrid_because_primack_is_jsonnet_disabled.json")

VOCABULARY_PAD_ID = 0

SAME_VARIABLE_TOKEN = '<IDENTITY>'
END_OF_VARIABLE_TOKEN = '</s>'

MINIMUM_TYPE_FREQUENCY = 100

FUNCTIONS_VOCAB_FILE_NAME = 'functions_vocab'
VOCAB_PICKLE_FILE_NAME = 'all_vocab_pkl'

UNCHANGED_VARIABLE_WEIGHT = 0.1

PROGRAM_GRAPH_FUNCTION_NODE_MANGLED_NAME = 'mangled_function_names'
PROGRAM_GRAPH_FUNCTION_NODE_DEMANGLED_NAME = 'label'
PROGRAM_GRAPH_FUNCTION_NODE_DEMANGLED_NAME_TOKENIZED = 'demangled_function_name_tokenized'
PROGRAM_GRAPH_FUNCTION_NODE_SHOULD_VALIDATE = 'should_validate'

GRAPH_ADDR_TO_FUNC_NAME_FILENAME_SUFFIX = "_addr_to_name.pkl"
DEMANGLED_NAME = 'demangled_name'
FUNCTION_DOESNT_APPEAR_IN_BOTH_GRAPHS_DEMANGLED_NAME_PLACEHOLDER = "NOPEITTYDOPE"

import sys

gettrace = getattr(sys, 'gettrace', None)

DEBUG_MODE = False


class EnvVars:
    use_gpu = None
    torch_device = None

    def change_use_gpu(self, value):
        self.use_gpu = value
        self.torch_device = "cuda" if self.use_gpu else "cpu"


env_vars = EnvVars()
if gettrace is None:
    print('release mode')
    env_vars.change_use_gpu(True)
elif gettrace():
    print('debug mode')
    env_vars.change_use_gpu(False)
    DEBUG_MODE = True
else:
    print('dunno if debug or release')
    env_vars.change_use_gpu(True)
NUM_WORKERS = 0
print(torch.cuda.memory_summary())
torch.cuda.empty_cache()
# NUM_WORKERS = 1

# env_vars.change_use_gpu(False)

MIN_SIZE_OF_LEGIT_SOFTWARE = 10

GENERIC_FUNCTION_NAME: str = 'subXXXXXX'

apt_lib_names_file_path: str = str(ROUTINIO_DIR.joinpath('DataAcquisition/letters/letter_{}_dbg_list.txt'))

HYBRID_PRETRAINED_MODEL_PATH = 'C:/Users/USER/PycharmProjects/routinio2.1/dire_neural_model/dire_neural_model/data/data/saved_models/model.hybrid.bin'

### Learning parameters ###
FUNCTION_NAME_DECODER_NUM_LAYERS = 1
FUNCTION_NAME_DECODER_DROPOUT = 0.1
ATTN_DECODER_RNN_MAX_LENGTH = 5
DATA_EXPLORATION_RESULTS = os.path.join(ROUTINIO_DIR, "experiments/data_exploration/results")
DATA_EXPLORATION_CACHE = os.path.join(ROUTINIO_DIR, "experiments/data_exploration/cache")

SLEEP_TIME = 1

STRIPPED = 'stripped'
ORIGINAL = 'original'

PROGRAM_NAME = 'program_name'
PROGRAM_ID = 'program_id'

EMBEDDING_SIZE = 200
pretrained_dire_model_path = '/home/jonathan/Desktop/Thesis/routinio2.1/dire_neural_model/dire_neural_model/data/data/saved_models/model.hybrid.bin'

RUN_START_DATE = str(date.today())[5:]

LOGGING_BASE_DIR = '/home/jonathan/Desktop/Thesis/routinio2.1/logs'
DIRE_OUTPUT_SIZE = 256
NERO_OUTPUT_SIZE = 512
SIZE_OF_VOCAB = 2000


