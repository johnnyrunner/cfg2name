from typing import List

from tqdm import tqdm

from config import *
from Datasets.dire_raw_dataset import DireRawFunctionData, DireRawSingleProgramDataset
from utils.general_utils import flatten_list
from vocabulary_builder.functions_vocab import FunctionsVocab
from vocabulary_builder.dire_variables_vocab import DireVariablesVocab


class FullVocab:
    @property
    def variables_dire_vocab(self) -> DireVariablesVocab:
        return self._variables_dire_vocab

    @property
    def functions_vocab(self) -> FunctionsVocab:
        return self._functions_vocab

    @functions_vocab.setter
    def functions_vocab(self, value):
        'setting'
        self._functions_vocab = value

    def __init__(self,
                 load_dire_from_files: bool = True,
                 raw_paths_list: List[List[DireRawFunctionData]] = None,
                 all_functions_names_list: List[str] = None,
                 dire_vocab_file=None,
                 functions_vocab_file=None,
                 functions_vocab_size=100):
        # TODO: currently only using
        self._dire_vocabulary_file_path = dire_vocab_file
        self._functions_vocabulary_file_path = functions_vocab_file
        # Load from list
        if load_dire_from_files:
            print('load variables dire vocab')
            self._variables_dire_vocab = DireVariablesVocab(load_from_existing_file=True,
                                                            vocab_file_prefix=self._dire_vocabulary_file_path)
        else:
            self._variables_dire_vocab = None
            print('The dire vocab did not load properly...')
            # print("Reading raw .jsonl files to build vocabulary")
            # programs_data = []
            # for raw_path in tqdm(raw_paths_list, unit='raw exe for dire vocabulary'):
            #     raw_dire_program_data_path = os.path.splitext(raw_path)[0] + DIRE_PROGRAM_DATA_FILE_EXTENSION
            #     programs_data.append(
            #         DireRawSingleProgramDataset.get_functions_data_from_json_path(raw_dire_program_data_path))
            # flattened_list_all_functions_raw = flatten_list(programs_data)
            # self._variables_dire_vocab = DireVariablesVocab(load_from_existing_file=False,
            #                                                 flattened_list_all_functions_raw=flattened_list_all_functions_raw,
            #                                                 vocab_file_prefix=self._dire_vocabulary_file_path)
        if all_functions_names_list is not None:
            print('generate new functions vocab')
            self._functions_vocab = FunctionsVocab(load_from_existing_file=False,
                                                   demangled_functions_names_list=all_functions_names_list,
                                                   vocab_file_prefix=self._functions_vocabulary_file_path,
                                                   vocabulary_size=functions_vocab_size)
        else:
            self._functions_vocab = None
