import hashlib

from shove import Shove

from torch import Tensor
from config import env_vars

BAD_EXAMPLES_HASHES = [
    '666a51026ca6d7fe5132a4c8c93807729cc88066547e4d8b07bc597e284db6ed',
    '70f18f315c80d8abd396b06d87cbcc8705f54f2834626d031a41364848658238',
    'a1bdf1b21728e95eab3b202690f03a3aead51f7784985d3fcc207e1d81139633',
    '904de0ebf903c84c0703c9af8f0b97609ddeaecaf67fabee8d37a99eb499a94c',
    '317d6f65cb61517f40def4135962b73ea3f250f2a0f21ce8f4882614c01dfa79',
    'a214eb23f0ade150a567aef3f7434edd03e03c4a397ba1564706e165279fc3c6',
    '7d3d9cf5f0a3409c6149af668fe8192d4d3f82adfbd1ef27fa959d786ce8c323',
    'af168d9ca45192139afce999a9bd9b6772e778c395a20bfb320ab293cb5c2fea',
    'eab25ee4f62759876d6b2ca8cb9dc08401e17c5e93bdff6e63fb997b61b5bc61',
    'c263095ca75c246a84cf9c28475c108ff710f9d6daf5efd48e1f1b318a9b1c94',
    '8063989279e42e67b50b1628bc1920aa7b815bcfc8872e6856f4da029e42d80c',
    '96160306669bbee2f1390d67866001165a8ebf99bb37bdd797e179b0c0748169',
    '147a1cf4367a6644a66ca6e4c14765148269bcaa99a17b0998f05fc393af8f1b',
    '8f4779248f693aa6d1341f6ed61e61ee61518215224557f8d02e8a6109d5180c',
    'f81439c2da8177a6d991a87a081ed3576d56dae5642cbe91150d027b360a4a74',
    'f603b201981c13475d6c4e43329307aa62409e78416e538e11018821ffb13f3c',
    '4776e28c1eb406ea6c02574b00967aa08580b36ac2543705dd5bd7faeff58181',
    '9644e5fb4ad31d26991946702238120d2ffcf6799e3dea4fdb3f6b9d270d7226',
    'ca1f6ad919a8d70c79a765308ab76dc2c825e7861a6a228361570ee1c817c020',
    '713915444ba73dcc19acb37ab4ea15dca1a0abd45243e70768ae20dd9b5d5662',
    'caaf0ec2edab2113458e223e7c66723c875247f75d3345718a3e7661bb247e30',
    'f3204b03c282e2420ae299fba67af71ab4ac6e6b0bf34a93fb2ed49e4026e369',
]


class DireCache:
    def __init__(self,
                 file_dict_location: str = 'file:///home/jonathan/Desktop/Thesis/routinio2.1/caches/shove_dict_cache'):
        self.examples = Shove(file_dict_location, sync=1)

    # def save_to_file(self, file_name: str = 'dire_cache.json'):
    #     examples = {}
    #     for k, v in self.examples.items():
    #         examples[k] = v
    #     with open(file_name, 'w+') as f:
    #         json.dump(examples, f)

    def hash_example(self, example, i=1):
        unique_value = example[i]['demangled_function_names']
        if i == 1:
            unique_value = unique_value[0]
        example_hash = hashlib.sha256(str(unique_value).encode('utf-8')).hexdigest()
        print(example_hash)
        return example_hash

    def insert_to_cache(self, example, dire_results):
        print(f'example:{example}')
        example_hash = self.hash_example(example)
        print(f'example_hash:{example_hash}')
        dire_results_0 = dire_results[0].tolist()
        dire_results_1 = dire_results[1]
        self.examples[example_hash] = (dire_results_0, dire_results_1)

    def is_inside(self, example):
        example_hash = self.hash_example(example)
        return example_hash in self.examples

    def is_hash_bad(self, example):
        example_hash = self.hash_example(example, i=0)
        print(example_hash)
        return example_hash in BAD_EXAMPLES_HASHES

    def get(self, example):
        example_hash = self.hash_example(example)
        print(example_hash)
        (dire_results_0, dire_results_1) = self.examples[example_hash]
        return (Tensor(dire_results_0).to(device=env_vars.torch_device), dire_results_1)

    # @staticmethod
    # def load(file_name: str = 'dire_cache.json'):
    #     returned_dire_cache = DireCache()
    #     with open(file_name, 'r') as f:
    #         saved_examples = json.load(f)
    #         for k, v in saved_examples.items():
    #             returned_dire_cache.examples[k] = v
    #     return returned_dire_cache


dire_cache = DireCache()
