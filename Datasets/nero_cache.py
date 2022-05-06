import hashlib

from shove import Shove

from torch import Tensor

from config import env_vars


class NeroCache:
    def __init__(self, file_dict_location: str = 'file:///home/jonathan/Desktop/Thesis/routinio2.1/caches/nero_cache'):
        self.examples = Shove(file_dict_location, sync=1)

nero_cache = NeroCache()
