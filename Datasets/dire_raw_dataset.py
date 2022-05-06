import json
import os

from config import DEMANGLED_NAME, logger
from dire_neural_model.utils.ast import *
from typing import List

import dire_neural_model.utils.ast
from dire_neural_model.utils.preprocess import generate_dire_example


class FunctionMetaData:

    @property
    def line_num(self) -> int:
        return self._line_num

    @property
    def file_name(self) -> str:
        return self._program_name

    def __init__(self, program_name, line_num):
        self._program_name = program_name
        self._line_num = line_num


class DireRawFunctionData:

    @property
    def ast(self) -> AbstractSyntaxTree:
        return self._ast

    @property
    def variable_name_map(self) -> dict:
        return self._variable_name_map

    @property
    def test_meta(self):
        return self._test_meta

    @property
    def code_tokens(self):
        return self._code_tokens

    @property
    def binary_file(self) -> FunctionMetaData:
        return self._function_metadata

    @property
    def demangled_name(self) -> str:
        return self._demangled_name

    @property
    def mangled_name(self) -> str:
        return self._ast.mangled_name

    def __init__(self, ast: AbstractSyntaxTree,
                 variable_name_map: dict,
                 function_metadata: FunctionMetaData,
                 demangled_name,
                 code_tokens = None,
                 test_meta = None):
        self._ast = ast
        self._variable_name_map = variable_name_map
        self._function_metadata = function_metadata
        self._code_tokens = code_tokens
        self._test_meta = test_meta
        self._demangled_name = demangled_name

    @classmethod
    def from_json_dict(cls, function_json_dict, function_metadata : FunctionMetaData):
        example, function_json_dict = generate_dire_example(function_json_dict)
        tree = example.ast
        variable_name_map = dict()

        for var_name, var_nodes in tree.variables.items():
            variable_name_map[var_name] = var_nodes[0].new_name

        # SP: Check if this is needed
        test_meta = None
        if 'test_meta' in function_json_dict:
            test_meta = function_json_dict['test_meta']

        # SP: Check if this is needed
        code_tokens = None
        if 'code_tokens' in function_json_dict:
            code_tokens = function_json_dict['code_tokens']

        raw_dire_function_data = cls(tree,
                                     variable_name_map,
                                     function_metadata,
                                     function_json_dict[DEMANGLED_NAME],
                                     code_tokens,
                                     test_meta)

        return raw_dire_function_data

class DireRawSingleProgramDataset:
    @staticmethod
    def get_functions_data_from_jsons_list(function_data_json_dicts, functions_data_json_path):
        function_metadatas = [FunctionMetaData(os.path.splitext(functions_data_json_path)[0], i)
                              for i in range(len(function_data_json_dicts))]
        lis = []
        for function_data_dict, function_metadata in zip(function_data_json_dicts, function_metadatas):
            try:
                lis.append(DireRawFunctionData.from_json_dict(function_data_dict, function_metadata))
            except:
                logger.info('failed to load function')
        return lis


    @staticmethod
    def get_functions_data_from_json_path(functions_data_json_path):
        function_data_json_dicts = [json.loads(function_line) for function_line in open(functions_data_json_path, 'r').readlines()]
        return DireRawSingleProgramDataset.get_functions_data_from_jsons_list(function_data_json_dicts, functions_data_json_path)

