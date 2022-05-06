import sys
import os
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from utils.files_utils import execute, switch_dir_to_ubuntu, switch_dir_to_windows

ARG_SCRIPT_FILE_NAME = 0
ARG_PROJECT_FOLDER_IDX = 1
ARG_RESULTS_FOLDER_IDX = 2
ARG_COUNT_MIN = 3
ARG_IS_IN_UBUNTU = 3

IS_UBUNTU = True

class ExecutableInfo:
    def __init__(self, path:str, build_id:str):
        self.path = path
        self.build_id = build_id

    def get_path(self) -> str :
        return self.path

    def get_build_id(self) -> str :
        return self.build_id

def path_by_os_type(path:str) -> str:
    return switch_dir_to_ubuntu(path) if IS_UBUNTU else switch_dir_to_windows(path)

def usage():
    print("Usage: <script name> <project debs folder> <results folder> <opt: is_ubuntu (true/false)>")

def extract_file_info_from_find_command_output(folder:str, find_command_output:str):
    AFTER_FILE_NAME_MARKER = ": ELF"
    BEFORE_BUILD_ID_MARKER = "BuildID[sha1]="
    AFTER_BUILD_ID_MARKER = ","

    file_path_relative = find_command_output[:find_command_output.find(AFTER_FILE_NAME_MARKER)]
    file_path_relative = file_path_relative.replace("./", "")
    file_path_absolute = os.path.join(folder if IS_UBUNTU else switch_dir_to_ubuntu(folder), file_path_relative)
    file_path_absolute = path_by_os_type(file_path_absolute)

    before_build_id_marker_idx = find_command_output.find(BEFORE_BUILD_ID_MARKER)
    build_id = find_command_output[before_build_id_marker_idx + len(BEFORE_BUILD_ID_MARKER): find_command_output.find(AFTER_BUILD_ID_MARKER, before_build_id_marker_idx)]

    return ExecutableInfo(file_path_absolute, build_id)

def get_all_executables_recursive(folder):
    executables_list = []
    print(f"Inside get_all_executables_recursive, folder={folder}")
    find_elfs_result = [i for i in execute((['ubuntu', 'run'] if not IS_UBUNTU else []) + ["find . -exec file {} \\; | grep -i elf"], folder)]
    for elf_result in find_elfs_result:
        executables_list.append(extract_file_info_from_find_command_output(folder, elf_result))

    return executables_list

def process_release_and_debug_files(results_folder:str, project_name:str, release_file_info : ExecutableInfo, debug_file_info : ExecutableInfo):
    results_with_symbols_path = os.path.join(results_folder, "binaries_with_symbols", project_name)
    results_without_symbols_path = os.path.join(results_folder, "binaries_without_symbols", project_name)
    Path(results_with_symbols_path).mkdir(exist_ok=True, parents=True)
    Path(results_without_symbols_path).mkdir(exist_ok=True, parents=True)

    base_name = os.path.basename(release_file_info.get_path())
    with_symbols_file_path = os.path.join(results_with_symbols_path, base_name + "_with_symbols")
    without_symbols_with_dynamic_file_path = os.path.join(results_without_symbols_path, base_name + "_without_symbols_with_dynamic")
    without_symbols_file_path = os.path.join(results_without_symbols_path, base_name + "_without_symbols")


    print()
    try:
        generate_full_elf_result = [i for i in execute((['ubuntu', 'run'] if not IS_UBUNTU else []) + [f"eu-unstrip --output={switch_dir_to_ubuntu(with_symbols_file_path)} \
        {switch_dir_to_ubuntu(release_file_info.get_path())} {switch_dir_to_ubuntu(debug_file_info.get_path())}"], results_folder)]

        strip_release_elf_result = [i for i in execute((['ubuntu', 'run'] if not IS_UBUNTU else []) + [f"strip --strip-all --output-file={switch_dir_to_ubuntu(without_symbols_with_dynamic_file_path)} \
         {switch_dir_to_ubuntu(release_file_info.get_path())}"], results_folder)]

        strip_dynsym_stripped_elf_result = [i for i in execute((['ubuntu', 'run'] if not IS_UBUNTU else []) + [f"strip -R .dynsym --output-file={switch_dir_to_ubuntu(without_symbols_file_path)} \
         {switch_dir_to_ubuntu(without_symbols_with_dynamic_file_path)}"], results_folder)]
    except:
        print(f'FAILED on {release_file_info}')

def main():
    if len(sys.argv) < ARG_COUNT_MIN:
        usage()
        exit(1)

    release_folder = os.path.join(sys.argv[ARG_PROJECT_FOLDER_IDX], "usr")
    debug_folder = os.path.join(sys.argv[ARG_PROJECT_FOLDER_IDX], "usr/lib/debug")
    results_folder = sys.argv[ARG_RESULTS_FOLDER_IDX]
    project_name = os.path.basename(sys.argv[ARG_PROJECT_FOLDER_IDX])
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)

    if len(sys.argv) > ARG_IS_IN_UBUNTU:
        if sys.argv[ARG_IS_IN_UBUNTU] == "true":
            IS_UBUNTU = True

    print(f"Project name: {project_name}")
    print(f"Release folder: {release_folder}")
    print(f"Debug folder: {debug_folder}")
    print(f"Results folder: {results_folder}")
    print(f"Is ubuntu: {IS_UBUNTU}")

    release_executables = [i for i in get_all_executables_recursive(release_folder) if not debug_folder in i.get_path()]
    debug_executables = get_all_executables_recursive(debug_folder)

    for i in range(len(release_executables)):
        for j in range(i + 1, len(release_executables)):
            if release_executables[i].get_build_id() == release_executables[j].get_build_id() or \
                os.path.basename(release_executables[i].get_path()) == os.path.basename(release_executables[j].get_path()):
                print(f"Found release duplicate:\r\n{release_executables[i].get_path()}, {release_executables[j].get_path()}\r\n{release_executables[i].get_build_id()}, {release_executables[j].get_build_id()}")

    for i in range(len(debug_executables)):
        for j in range(i + 1, len(debug_executables)):
            if debug_executables[i].get_build_id() == debug_executables[j].get_build_id() or \
                os.path.basename(debug_executables[i].get_path()) == os.path.basename(debug_executables[j].get_path()):
                print(f"Found debug duplicate:\r\n{debug_executables[i].get_path()}, {debug_executables[j].get_path()}\r\n{debug_executables[i].get_build_id()}, {debug_executables[j].get_build_id()}")

    for release_file_info in release_executables:
        for debug_file_info in debug_executables:
            if debug_file_info.get_build_id() == release_file_info.get_build_id():
                #if debug_file_info.get_build_id() != release_file_info.get_build_id() or os.path.basename(debug_file_info.get_path()) != os.path.basename(release_file_info.get_path()):
                #    print(f"Found partial mismatch:\r\n{release_file_info.get_path()}, {debug_file_info.get_path()}\r\n{release_file_info.get_build_id()}, {debug_file_info.get_build_id()}")
                #    continue
                # Match either in build id or name
                print(f"Processing files:{release_file_info.get_path()}, {debug_file_info.get_path()}")
                process_release_and_debug_files(results_folder, project_name, release_file_info, debug_file_info)


if __name__ == "__main__":
    main()