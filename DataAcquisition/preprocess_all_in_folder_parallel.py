import concurrent
import os
from pathlib import Path

from tqdm import tqdm

from DatasetGeneration.preprocess_binaries import preprocess_all_binaries_in_folder
from config import apt_lib_names_file_path, ALL_DATASETS_DIR, DECOMPILED_BINARIES_DATA_DIR
from utils.files_utils import execute
from utils.function_utils import from_debug_name_to_name

all_dataset_lib_folder = 'C:\\Users\\USER\\Desktop\\University\\RE\\datasets\\code_and_build_artifacts\\code_binaries\\code_binaries\\data'
one_lib_folder = 'C:\\Users\\USER\\Desktop\\University\\RE\\datasets\\code_and_build_artifacts\\code_binaries\\code_binaries\\data\\MDEwOlJlcG9zaXRvcnk0MDE5NzEwMA==\\derivatives\\gonmf-sudoku-solver-b6197a9'
another_lib_folder = 'C:\\Users\\USER\\Desktop\\University\\RE\\datasets\\code_and_build_artifacts\\code_binaries\\code_binaries\\data\\MDEwOlJlcG9zaXRvcnk0NTYwNTI1MQ==\\derivatives\\regehr-fs-fuzz-514a2f5'
another_one_lib_folder = 'C:\\Users\\USER\\Desktop\\University\\RE\\datasets\\code_and_build_artifacts\\code_binaries\\code_binaries\\data\\1example'

if __name__ == '__main__':
    # before running this script one whould run using ubuntu the script
    # C:\\Users\\USER\\PycharmProjects\\routinio2.1\\DataAcquisition\\download_list_of_dbg_available_libs.sh
    # split_apt_names_by_first_char(all_apt_lib_names)

    # data_path = os.path.join(ALL_OUTPUTS_DIR, "binaries_with_symbols")
    # preprocess_all_binaries_in_folder(all_dataset_lib_folder, DECOMPILED_BINARIES_DATA_DIR, num_parallel_threads=8,
    #                                   stripped_binaries_dir=None, use_stripped_path=False)
    # preprocess_all_binaries_in_folder(another_one_lib_folder, DECOMPILED_BINARIES_DATA_DIR, num_parallel_threads=1,
    #                                   stripped_binaries_dir=None, use_stripped_path=False)
    preprocess_all_binaries_in_folder(all_dataset_lib_folder, DECOMPILED_BINARIES_DATA_DIR, num_parallel_threads=1,
                                      stripped_binaries_dir=None, use_stripped_path=False)

