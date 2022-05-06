from unittest import TestCase


from config import ROUTINIO_DIR
from Datasets.dire_raw_dataset import DireRawSingleProgramDataset
from utils.graph_utils import MergeFunctionsDataIntoNetworkx
from vocabulary_builder.dire_variables_vocab import DireVariablesVocab



class TestVariablesDireVocab(TestCase):
    def test_add_from_dataset(self):
        test_dataset_root = ROUTINIO_DIR.joinpath('data/networkx_test')
        test_dataset_root.joinpath('raw').mkdir(parents=True, exist_ok=True)

        networkx_examples = MergeFunctionsDataIntoNetworkx(ROUTINIO_DIR.joinpath('data/decompiled_binaries'),
                                                           test_dataset_root.joinpath('raw'))

        programs_dire_list = DireRawSingleProgramDataset.get_functions_data_from_jsonls(test_dataset_root)

        vocab_builder = DireVariablesVocab()
        vocab_builder.add_from_list_of_lists(program_dataset)