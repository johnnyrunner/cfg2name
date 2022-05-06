from collections import Counter
from typing import List

from config import VOCABULARY_PAD_ID

import sentencepiece as spm

from dire_neural_model.utils.dire_vocab import VocabEntry


class FunctionsVocab:
    def __init__(self,
                 load_from_existing_file: bool = True,
                 demangled_functions_names_list: List[str] = None,
                 vocab_file_prefix=None,
                 vocabulary_size=100,
                 character_coverage=0.9995):

        self._vocabulary_file_name_prefix = vocab_file_prefix
        self._character_coverage = character_coverage
        if load_from_existing_file and vocab_file_prefix:
            self._vocab_entry = VocabEntry.load(vocab_file_prefix)
            self._vocabulary_size = len(self._vocab_entry.id2word.keys())
        else:
            self._vocabulary_size = vocabulary_size
            self._vocab_entry = self.build_vocab_entry_from_functions_data(demangled_functions_names_list)

    def eos_id(self):
        return self._vocab_entry.eos_id()

    def bos_id(self):
        return self._vocab_entry.bos_id()

    def encode_as_subtoken_ids(self, word: str):
        return self._vocab_entry.vectorized_encode_as_subtoken_ids(word)

    def normal_encode_as_subtoken_ids(self, word: str):
        return self._vocab_entry.encode_as_subtoken_ids_list(word)

    def encode_as_subtokens_list(self, word:str):
        return [self._vocab_entry.id2word[subtoken_id] for subtoken_id in self.normal_encode_as_subtoken_ids(word)]

    def encode_as_subtokens(self, word: str):
        return ','.join(self.encode_as_subtokens_list(word))

    def get_id2word(self):
        return {v: k for k, v in self._vocab_entry.word2id.items()}

    def build_vocab_entry_from_functions_data(self, demangled_functions_names_list: List[str]):
        print('building target words vocabulary')
        tgt_func_names_file = self._vocabulary_file_name_prefix + '.func_names.txt'
        with open(tgt_func_names_file, 'w') as f:
            for index, name in demangled_functions_names_list:
                f.write(name + '\n')
        # spm.SentencePieceTrainer.Train(
        #     f'--add_dummy_prefix=false --pad_id={VOCABULARY_PAD_ID} --bos_id=1 --eos_id=2 --unk_id=3 '
        #     f'--control_symbols=<IDENTITY> '
        #     # f'--character_coverage={selafl mutation strategiesf._character_coverage}'
        #     f'--vocab_size={self._vocabulary_size} '
        #     f'--model_prefix="{self._vocabulary_file_name_prefix}.tgt" --model_type=bpe '
        #     f'--input="{tgt_func_names_file}"')
        spm.SentencePieceTrainer.Train(
            input=[tgt_func_names_file],
            model_prefix=f"{self._vocabulary_file_name_prefix}.tgt",
            model_type='bpe',
            vocab_size=self._vocabulary_size,
            character_coverage=self._character_coverage,
            control_symbols='<IDENTITY>',
            add_dummy_prefix='false',
            pad_id=VOCABULARY_PAD_ID,
            bos_id='1',
            eos_id='2',
            unk_id='3'
        )
        tgt_functions_vocab_entry = VocabEntry(''+self._vocabulary_file_name_prefix + '.tgt.model',)
                                               # vocab_size=self._vocabulary_size)
        tgt_functions_vocab_entry.save(self._vocabulary_file_name_prefix)

        names_counts = Counter(demangled_functions_names_list)
        names_and_count = [(f'{key[1]} : {value}', value) for key, value in names_counts.items()]
        with open(self._vocabulary_file_name_prefix + '.wordslist.txt', 'w+') as f:
            for final_string_for_item, count in sorted(names_and_count, key=lambda x: x[1], reverse=True):
                f.write('%s \n' % final_string_for_item)

        print('printed function names to file')

        with open(self._vocabulary_file_name_prefix + '.words_stats.txt', 'w+') as f:
            print(f'total number of functions {sum([key for value, key in names_and_count])}\n', file=f)
            print(f'total number of function distinct names {len(names_and_count)}\n', file=f)
            print(f'sum of sub starting functions occurences {sum([key for value, key in names_and_count if value[:3] =="sub"])}\n', file=f)
            print(f'number of sub starting functions {len([key for value, key in names_and_count if value[:3] =="sub"])}\n', file=f)

        print('printed words stats names to file')

        return tgt_functions_vocab_entry
