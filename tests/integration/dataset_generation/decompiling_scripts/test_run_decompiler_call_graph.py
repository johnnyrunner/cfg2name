from unittest import TestCase

from DatasetGeneration.decompiler_scripts.features_generator import FeatureGenerator
from config import DECOMPILED_BINARIES_TEST_DIR, BINARIES_DIR



class Test(TestCase):
    def test_extract_call_graph(self):
        binary = 'binary_search.o'
        features_generator = FeatureGenerator(DECOMPILED_BINARIES_TEST_DIR)
        file_path, _ = features_generator.configure_for_ida_call(binary)
        features_generator.extract_call_graph(binary, file_path)
