from unittest import TestCase

from utils.general_utils import flatten_list


class Test(TestCase):
    def test_flatten_list(self):
        flattened = flatten_list([[1, 2], [3, 4]])
        self.assertListEqual(flattened, [1, 2, 3, 4])
