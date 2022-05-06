from torch_geometric.data import Batch

from Datasets.programs_dataset import ProgramsDataset
from NeuralModels.function_namer import FunctionNamer
from config import ROUTINIO_DIR

PATH_TO_CHECKPOINT = 'C:\\Users\\USER\\PycharmProjects\\routinio2.1\\experiments\\lightning_logs_18.7\\version_115\\checkpoints\\epoch=9.ckpt'

PROGRAM_GRAPH_DATASET_ROOT_DIR_5_SMALL = ROUTINIO_DIR.joinpath('data/5_basic_functions_dataset')
PROGRAM_GRAPH_DATASET_ROOT_DIR_5_SMALL.mkdir(parents=True, exist_ok=True)
programs_dataset = ProgramsDataset(PROGRAM_GRAPH_DATASET_ROOT_DIR_5_SMALL, load_dire_vocab_from_file=True)

model = FunctionNamer.load_from_checkpoint(checkpoint_path=PATH_TO_CHECKPOINT)
model.eval()

data_list = [x for x in iter(programs_dataset)]
batch = Batch.from_data_list([data_list[0]])
y_hat = model(batch)

