import errno
import logging
import os
import pathlib
import subprocess
import tempfile
from pathlib import Path

from config import ORIGINAL_BINRARIES_DIRS, logger, PROGRAM_GRAPH_DATASET_ROOT_DIR


def execute(cmd, sub_dir):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                             universal_newlines=True, shell=True,
                             cwd=sub_dir, stderr=subprocess.STDOUT)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def make_directory(dir_path):
    """Make a directory, with clean error messages."""
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if not os.path.isdir(dir_path):
            raise NotADirectoryError(f"'{dir_path}' is not a directory")
        if e.errno != errno.EEXIST:
            raise


def switch_dir_to_ubuntu(directory:str):
    if directory.startswith("/mnt"):
        return directory
    if directory.startswith('C'):
        return '/mnt/c/' + directory.replace('\\', '/').replace('C:/', '')
    elif directory.startswith('D'):
        return '/mnt/d/' + directory.replace('\\', '/').replace('D:/', '')
    assert False


def switch_dir_to_windows(directory:str):
    if directory.startswith("C:") or directory.startswith("D:"):
        return directory
    if directory.startswith('/mnt/c'):
        return 'C:' + directory.replace('/mnt/c', '').replace('/', '\\')
    elif directory.startswith('/mnt/d'):
        return 'C:' + directory.replace('/mnt/d', '').replace('/', '\\')
    assert False


def create_temp_copy(file_path):
    # Create a temporary directory, since the decompiler makes a lot of additional
    # files that we can't clean up from here
    # with tempfile.TemporaryDirectory() as tempdir:
    # tempfile.tempdir = tempdir
    # tempdir = C:\Users\USER\Desktop\Thesis\routinio2\data\tmp #tempfile.TemporaryDirectory()
    # tempfile.tempdir = tempdir.name
    # File counts for progress output
    file_copy = tempfile.NamedTemporaryFile(delete=False)
    file_copy_name = file_copy.name
    file_copy.close()
    src = open(file_path, 'rb')
    file_copy = open(file_copy_name, 'wb')
    file_copy.write(src.read())
    file_copy.close()
    src.close()
    return file_copy_name

def inplace_change_word_occurences_in_file(filename:str, old_string:str, new_string:str):
    # Safely read the input filename using 'with'
    with open(filename) as f:
        s = f.read()
        if old_string not in s:
            logger.info('"{old_string}" not found in {filename}.'.format(**locals()))
            return

    # TODO: this is not safe, if there are different things called the same as the function it does not work!
    # Safely write the changed content, if found in the file
    with open(filename, 'w') as f:
        logger.info('Changing "{old_string}" to "{new_string}" in {filename}'.format(**locals()))
        s = s.replace(old_string, new_string)
        f.write(s)


def get_program_id_from_gexf_name(gexf_file_name, gexf_files_dir):
    program_name = '.'.join(gexf_file_name.split('.')[:-1])
    decompiled_db_path = Path(gexf_files_dir)
    program_name_to_program_original_dir = os.path.join(
        os.path.join(decompiled_db_path.parent.absolute(), ORIGINAL_BINRARIES_DIRS), program_name + '.txt')
    with open(program_name_to_program_original_dir, 'r') as f:
        program_original_dir = f.read()
    program_id = program_original_dir.split('data\\')[-1].split('\\')[0]
    return program_id


def delete_dataset_if_needed(reload_dataset, dataset_root_dir=PROGRAM_GRAPH_DATASET_ROOT_DIR):
    if reload_dataset:
        for file_path in pathlib.Path(dataset_root_dir).glob(f"**/*"):
            if os.path.isfile(file_path):
                os.remove(file_path)