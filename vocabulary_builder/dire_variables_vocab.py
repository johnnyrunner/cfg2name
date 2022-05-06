from collections import Counter
from typing import List

from config import DIRE_VOCABULARY_EXAMPLE_FILE, VOCABULARY_PAD_ID, SAME_VARIABLE_TOKEN, MINIMUM_TYPE_FREQUENCY
from Datasets.dire_raw_dataset import DireRawFunctionData
from dire_neural_model.utils.grammar import Grammar
from dire_neural_model.utils.dire_vocab import VocabEntry, DireVocab
import sentencepiece as spm

from utils.general_utils import flatten_list


class DireVariablesVocab:

    @property
    def dire_vocab(self) -> DireVocab:
        return self._dire_vocab

    @property
    def vocabulary_file_path(self):
        return self._vocabulary_file_name_prefix

    def __init__(self,
                 load_from_existing_file: bool = True,
                 flattened_list_all_functions_raw: List[DireRawFunctionData] = None,
                 vocab_file_prefix=None,
                 size: int = 10000,
                 freq_cutoff=2):

        self._vocabulary_size = size
        self._vocabulary_file_name_prefix = vocab_file_prefix
        self._freq_cutoff = freq_cutoff

        # Build vocab from loaded data
        if not load_from_existing_file and flattened_list_all_functions_raw is not None:
            self._dire_vocab = self.build_dire_vocab_from_list_of_lists(flattened_list_all_functions_raw)

        else:
            self._dire_vocab = DireVocab.load(self._vocabulary_file_name_prefix)

    def build_dire_vocab_from_list_of_lists(self, flattened_list_all_functions_raw: List[DireRawFunctionData]) -> DireVocab:
        src_code_tokens_dir = self._vocabulary_file_name_prefix + '.src_code_tokens.txt'

        identifier_names, node_types, src_preserved_tokens, src_words, tgt_words, \
        type_tokens = self.get_tokenization_data_for_dire(flattened_list_all_functions_raw, src_code_tokens_dir)

        print('building source words vocabulary')
        src_var_vocab_entry = VocabEntry.from_corpus([src_words], size=self._vocabulary_size,
                                                     freq_cutoff=self._freq_cutoff)

        src_code_tokens_vocab_entry, tgt_var_vocab_entry = self.build_bpe_src_and_tgt_var_vocab(
            src_code_tokens_dir, src_preserved_tokens, tgt_words)

        id_names_file = self._vocabulary_file_name_prefix + '.id_names.txt'
        with open(id_names_file, 'w') as f:
            for name in identifier_names:
                f.write(name + '\n')

        print('train subtoken model for obj names')
        # train subtoken models
        spm.SentencePieceTrainer.Train(
            f'--add_dummy_prefix=false --pad_id={VOCABULARY_PAD_ID} --bos_id=1 --eos_id=2 --unk_id=3 '
            f'--control_symbols=<IDENTITY> --vocab_size={self._vocabulary_size} '
            f'--model_prefix={self._vocabulary_file_name_prefix}.obj_name --model_type=bpe '
            f'--input={id_names_file} '
            f'--hard_vocab_limit=false'  # TODO: DELETE hard_vocab_limit
        )
        obj_name_vocab_entry = VocabEntry(self._vocabulary_file_name_prefix + '.obj_name.model')

        type_vocab = Counter(type_tokens)
        var_types = []
        for type_token, freq in type_vocab.items():
            if freq > MINIMUM_TYPE_FREQUENCY:
                print(type_token, freq)
                var_types.append(type_token)

        print('init node types and variable types')
        grammar = Grammar(node_types, var_types)

        print('Node types:', node_types)
        print('Variable types:', var_types)

        vocab = DireVocab(source=src_var_vocab_entry,
                      source_tokens=src_code_tokens_vocab_entry,
                      target=tgt_var_vocab_entry,
                      obj_name=obj_name_vocab_entry,
                      grammar=grammar)

        vocab.save(self._vocabulary_file_name_prefix)
        return vocab

    def get_tokenization_data_for_dire(self, dire_data, src_code_tokens_file):
        src_preserved_tokens = set()
        f_src_token = open(src_code_tokens_file, 'w')
        # extract vocab and node types
        node_types = set()
        src_words = []
        tgt_words = []
        identifier_names = []
        type_tokens = []
        for function_example in dire_data:
            for node in function_example.ast:
                node_types.add(node.node_type)

                if node.is_variable_node:
                    old_var_name = node.old_name
                    new_var_name = node.new_name

                    src_words.append(old_var_name)

                    if old_var_name != new_var_name:
                        tgt_words.append(new_var_name)

                if node.node_type == 'obj' or node.node_type == 'block' and hasattr(node, 'name'):
                    identifier_names.append(node.name)

                if hasattr(node, 'type_tokens'):
                    type_tokens.extend(node.type_tokens)

            code_tokens = function_example.code_tokens
            preserved_tokens = [token for token in code_tokens if token.startswith('@@') and token.endswith('@@')]
            src_preserved_tokens.update(preserved_tokens)
            f_src_token.write(' '.join(code_tokens) + '\n')
        f_src_token.close()
        return identifier_names, node_types, src_preserved_tokens, src_words, tgt_words, type_tokens

    def build_bpe_src_and_tgt_var_vocab(self, src_code_tokens_file, src_preserved_tokens, tgt_words):
        print('use bpe')
        print('building source code tokens vocabulary')
        # train subtoken models
        src_preserved_tokens = ','.join(src_preserved_tokens)
        spm.SentencePieceTrainer.Train(
            f'--add_dummy_prefix=false --pad_id={VOCABULARY_PAD_ID} --bos_id=1 --eos_id=2 --unk_id=3 '
            f'--user_defined_symbols={src_preserved_tokens} '
            f'--vocab_size={self._vocabulary_size} '
            f'--model_prefix={self._vocabulary_file_name_prefix}.src_code_tokens --model_type=bpe '
            f'--input={src_code_tokens_file} '
            f'--hard_vocab_limit=false' #TODO: DELETE hard_vocab_limit
        )
        src_code_tokens_vocab_entry = VocabEntry(self._vocabulary_file_name_prefix + '.src_code_tokens.model')

        print('building target words vocabulary')
        tgt_word_file = self._vocabulary_file_name_prefix + '.var_names.txt'
        with open(tgt_word_file, 'w') as f:
            for name in tgt_words:
                f.write(name + '\n')
        spm.SentencePieceTrainer.Train(
            f'--add_dummy_prefix=false --pad_id={VOCABULARY_PAD_ID} --bos_id=1 --eos_id=2 --unk_id=3 '
            f'--control_symbols=<IDENTITY> '
            f'--vocab_size={self._vocabulary_size} '
            f'--model_prefix={self._vocabulary_file_name_prefix}.tgt --model_type=bpe '
            f'--input={tgt_word_file} '
            f'--hard_vocab_limit=false' #TODO: DELETE hard_vocab_limit
        )
        tgt_var_vocab_entry = VocabEntry(self._vocabulary_file_name_prefix + '.tgt.model')
        return src_code_tokens_vocab_entry, tgt_var_vocab_entry
