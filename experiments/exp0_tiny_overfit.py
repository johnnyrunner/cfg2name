# the 1000 words vocab# dire pretrained model location = HYBRID_PRETRAINED_MODEL_PATH


from pytorch_lightning import Trainer

from NeuralModels.function_namer import FunctionNamer
from NeuralModels.programs_data_module import ProgramsDataModule
from config import *
from Datasets.programs_dataset import ProgramsDataset
from utils.general_utils import set_seed

EMBEDDING_SIZE = 32

SEED = 44
set_seed(SEED)

TINY_PROGRAM_GRAPH_DATASET_ROOT_DIR = os.path.join(ALL_DATASETS_DIR, 'tiny_final_stage_dataset')
PROGRAM_GRAPH_DATASET_RAW_DIR_TINY, PROGRAM_GRAPH_DATASET_PROCESSED_DIR_TINY = make_dataset_dir(
    TINY_PROGRAM_GRAPH_DATASET_ROOT_DIR)

small_programs_dataset = ProgramsDataset(TINY_PROGRAM_GRAPH_DATASET_ROOT_DIR, load_dire_vocab_from_file=True)


programs_datamodule = ProgramsDataModule(small_programs_dataset, 50, 25, 1)
functions_vocabulary = small_programs_dataset.functions_vocab

print("Creating Model...")
# dire pretrained model location = HYBRID_PRETRAINED_MODEL_PATH
model = FunctionNamer(EMBEDDING_SIZE, functions_vocabulary)
if env_vars.use_gpu:
    trainer = Trainer(gpus=1)
else:
    trainer = Trainer()

print("Training Model...")
trainer.fit(model, programs_datamodule)
