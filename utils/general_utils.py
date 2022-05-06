import random
from itertools import compress

import numpy
import torch
from torch import Tensor


def flatten_list(list_of_list):
    return [item for item_list in list_of_list for item in item_list]


def set_seed(seed):
    torch.random.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)

def lists_are_equal(list_1, list_2):
    for a in list_1:
        if a not in list_2:
            return False
    for a in list_2:
        if a not in list_1:
            return False
    return True

def get_masked_iterable(iterable, mask):
    if type(iterable) == list:
        mask = [1 if (i in mask) else 0 for i in range(len(iterable))]
        return list(compress(iterable, mask))
    elif type(iterable) == Tensor:
        return iterable[mask.type(torch.LongTensor)]
    print(type(iterable))
    return None