import pathlib
import shutil
import random

import numpy
import torch
from pytorch_lightning import Trainer

from NeuralModels.function_namer import FunctionNamer
from NeuralModels.programs_data_module import ProgramsDataModule
from config import *
from Datasets.programs_dataset import ProgramsDataset
from utils.graph_utils import MergeFunctionsDataIntoNetworkx


import os

from utils.general_utils import set_seed
from utils.graph_utils import MergeFunctionsDataIntoNetworkx

RELOAD_DATASET = True
if RELOAD_DATASET:
    for file_path in pathlib.Path(PROGRAM_GRAPH_DATASET_ROOT_DIR).glob(f"**/*"):
        if os.path.isfile(file_path):
            os.remove(file_path)
print("Merging data from function call graph into dire AST data, and moving files into raw folder...")
MergeFunctionsDataIntoNetworkx(DECOMPILED_BINARIES_DATA_DIR, PROGRAM_GRAPH_DATASET_RAW_DIR)
EMBEDDING_SIZE = 32


SEED = 44
set_seed(SEED)
print("Building ProgramsDataModule...")
programs_dataset = ProgramsDataset(PROGRAM_GRAPH_DATASET_ROOT_DIR, load_dire_vocab_from_file=True,
                                   functions_vocab_size=1000, stripped_or_original=ORIGINAL)
programs_datamodule = ProgramsDataModule(programs_dataset, 50, 25, 1)
functions_vocabulary = programs_dataset.functions_vocab

print("Creating Model...")
# dire pretrained model location = HYBRID_PRETRAINED_MODEL_PATH
model = FunctionNamer(EMBEDDING_SIZE, functions_vocabulary)
if env_vars.use_gpu:
    trainer = Trainer(gpus=1)
else:
    trainer = Trainer()


print("Training Model...")
trainer.fit(model, programs_datamodule)
