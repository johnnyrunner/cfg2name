from unittest import TestCase

from DatasetGeneration.decompiler_scripts.features_generator import FeatureGenerator
from config import DECOMPILED_BINARIES_TEST_DIR, BINARIES_DIR


class Test(TestCase):
    def test_extract_features_for_binary(self):
        binary = 'binary_search.o'
        features_generator = FeatureGenerator(DECOMPILED_BINARIES_TEST_DIR)#, ida_version=IDA32_PATH)
        features_generator.extract_features_for_binary(binary)