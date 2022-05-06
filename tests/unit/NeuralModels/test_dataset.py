from unittest import TestCase

from DatasetGeneration.dataset import SimpleDataModule
from config import ROUTINIO_DIR, YES

class TestSimpleDataModule(TestCase):
    def test_setup(self):
        data_loader = SimpleDataModule(ROUTINIO_DIR.joinpath('data//decompiled_binaries_test'))
        data_loader.setup(YES)