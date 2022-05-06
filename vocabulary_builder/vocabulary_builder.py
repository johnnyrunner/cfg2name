from collections import Counter

from config import DIRE_VOCABULARY_EXAMPLE_FILE, VOCABULARY_PAD_ID, SAME_VARIABLE_TOKEN, MINIMUM_TYPE_FREQUENCY
from dire_neural_model.utils.grammar import Grammar
from dire_neural_model.utils.dire_vocab import *
import sentencepiece as spm



class VocabularyBuilder:
    def __init__(self, size: int = 100, vocab_file=DIRE_VOCABULARY_EXAMPLE_FILE, freq_cutoff=2, use_bpe=True):
        self.vocabulary_size = size
        self.vocabulary_file = vocab_file
        self.freq_cutoff = freq_cutoff
        self.use_bpe = use_bpe

    def add_from_dataset(self, dataset):
        for program_example, y in dataset:
            graph_data, dire_data = program_example
            from dire_neural_model.utils.dataset import Dataset
            src_code_tokens_file = self.vocabulary_file + '.src_code_tokens.txt'

            identifier_names, node_types, src_preserved_tokens, src_words, tgt_words,\
            type_tokens = self.get_tokenization_data_for_dire(dire_data, src_code_tokens_file)

            print('building source words vocabulary')
            src_var_vocab_entry = VocabEntry.from_corpus([src_words], size=self.vocabulary_size,
                                                         freq_cutoff=self.freq_cutoff)

            if self.use_bpe:
                src_code_tokens_vocab_entry, tgt_var_vocab_entry = self.build_bpe_src_and_tgt_var_vocab(
                    src_code_tokens_file, src_preserved_tokens, tgt_words)
            else:
                tgt_var_vocab_entry = VocabEntry.from_corpus([tgt_words], size=self.vocabulary_size,
                                                             freq_cutoff=int(self.freq_cutoff),
                                                             predefined_tokens=[SAME_VARIABLE_TOKEN])

            id_names_file = vocab_file + '.id_names.txt'
            with open(id_names_file, 'w') as f:
                for name in identifier_names:
                    f.write(name + '\n')

            print('train subtoken model for obj names')
            # train subtoken models
            spm.SentencePieceTrainer.Train(
                f'--add_dummy_prefix=false --pad_id={VOCABULARY_PAD_ID} --bos_id=1 --eos_id=2 --unk_id=3 '
                f'--control_symbols=<IDENTITY> --vocab_size={self.vocabulary_size} '
                f'--model_prefix={self.vocabulary_file}.obj_name --model_type=bpe '
                f'--input={id_names_file}')
            obj_name_vocab_entry = VocabEntry(vocab_file + '.obj_name.model')

            type_vocab = Counter(type_tokens)
            num_types = 100
            var_types = []
            for type_token, freq in type_vocab.items():
                if freq > MINIMUM_TYPE_FREQUENCY:
                    print(type_token, freq)
                    var_types.append(type_token)

            print('init node types and variable types')
            grammar = Grammar(node_types, var_types)

            print('Node types:', node_types)
            print('Variable types:', var_types)

            vocab = Vocab(source=src_var_vocab_entry,
                          source_tokens=src_code_tokens_vocab_entry,
                          target=tgt_var_vocab_entry,
                          obj_name=obj_name_vocab_entry,
                          grammar=grammar)

            vocab.save(self.vocabulary_file)

    def get_tokenization_data_for_dire(self, dire_data, src_code_tokens_file):
        src_preserved_tokens = set()
        f_src_token = open(src_code_tokens_file, 'w')
        # extract vocab and node types
        node_types = set()
        src_words = []
        tgt_words = []
        identifier_names = []
        type_tokens = []
        for program_example in dire_data:
            for node in program_example.ast:
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

            code_tokens = program_example.code_tokens
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
            f'--vocab_size={self.vocabulary_size} '
            f'--model_prefix={self.vocabulary_file}.src_code_tokens --model_type=bpe '
            f'--input={src_code_tokens_file}')
        src_code_tokens_vocab_entry = VocabEntry(vocab_file + '.src_code_tokens.model')

        print('building target words vocabulary')
        tgt_word_file = self.vocabulary_file + '.var_names.txt'
        with open(tgt_word_file, 'w') as f:
            for name in tgt_words:
                f.write(name + '\n')
        spm.SentencePieceTrainer.Train(
            f'--add_dummy_prefix=false --pad_id={VOCABULARY_PAD_ID} --bos_id=1 --eos_id=2 --unk_id=3 '
            f'--control_symbols=<IDENTITY> '
            f'--vocab_size={self.vocabulary_size} '
            f'--model_prefix={self.vocabulary_file}.tgt --model_type=bpe '
            f'--input={tgt_word_file}')
        tgt_var_vocab_entry = VocabEntry(vocab_file + '.tgt.model')
        return src_code_tokens_vocab_entry, tgt_var_vocab_entry
