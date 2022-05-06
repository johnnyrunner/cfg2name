import os
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from DatasetGeneration.preprocess_binaries import preprocess_all_binaries_in_folder
from config import apt_lib_names_file_path, ALL_DATASETS_DIR, DECOMPILED_BINARIES_DATA_DIR
from utils.files_utils import execute
from utils.function_utils import from_debug_name_to_name

all_apt_lib_names = 'C:\\Users\\USER\\PycharmProjects\\routinio2.1\\DataAcquisition\\all_dbg_libs.txt'


def split_apt_names_by_first_char(all_apt_lib_names_file_path: str):
    lists_path = 'C:\\Users\\USER\\PycharmProjects\\routinio2.1\\DataAcquisition\\letters'
    lists_path_obj = Path(lists_path)
    lists_path_obj.mkdir(exist_ok=True, parents=True)
    dictionary_by_first_letter = {}
    for name in open(all_apt_lib_names_file_path, 'r').readlines():
        if name[0] in dictionary_by_first_letter:
            dictionary_by_first_letter[name[0]].append(name)
        else:
            dictionary_by_first_letter[name[0]] = [name]

    for letter, names_list in dictionary_by_first_letter.items():
        path = lists_path_obj / f'letter_{letter}_dbg_list.txt'
        with open(path, 'w+') as f:
            for name in names_list:
                f.write("%s" % name)



def main(apt_lib_names_file_path, preprocess_after_download: bool = False, num_parallel_threads=4):
    print('started downloading $ preprocessing')
    all_dbg_libs_names = open(apt_lib_names_file_path, 'r')
    all_dbg_libs_names_list = all_dbg_libs_names.readlines()
    dir_name = os.path.dirname(all_apt_lib_names)
    if preprocess_after_download:
        executor = ThreadPoolExecutor(max_workers=num_parallel_threads)

    for name in tqdm(all_dbg_libs_names_list):
        orig_lib_name = from_debug_name_to_name(name.strip())
        print(f"name: {name} original name: {orig_lib_name}")
        if orig_lib_name not in os.listdir(os.path.join(ALL_DATASETS_DIR, 'apt_related\\apt-debs')):
            try:
                for i in execute(
                        ['ubuntu', 'run', 'echo 123qweasd|sudo -S', 'bash', "download_file_and_dbg", orig_lib_name],
                        sub_dir=dir_name):
                    print(i)
            except:
                print('FAILED in files things')
            if preprocess_after_download:
                executor.submit(preprocess_all_binaries_in_folder, os.path.join(ALL_DATASETS_DIR, "binaries_with_symbols", orig_lib_name), DECOMPILED_BINARIES_DATA_DIR)
                print('sent thread')

    print('finished downloading')



if __name__ == '__main__':
    # before running this script one whould run using ubuntu the script
    # C:\\Users\\USER\\PycharmProjects\\routinio2.1\\DataAcquisition\\download_list_of_dbg_available_libs.sh
    # split_apt_names_by_first_char(all_apt_lib_names)
    main(all_apt_lib_names, preprocess_after_download=True, num_parallel_threads=8)
