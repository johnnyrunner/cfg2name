from NeuralModels.function_namer import FunctionNamer
from NeuralModels.sub_models import TopNeuralModels
from config import *
from config import DatasetDirs

from experiments.experiments_utils import prepare_default_experiment
from utils.files_utils import delete_dataset_if_needed
from utils.graph_utils import MergeFunctionsDataIntoNetworkxSMALL_PROGRAM_GRAPH_DATASET_ROOT_DIR

delete_dataset_if_needed(reload_dataset=False)
MergeFunctionsDataIntoNetworkx(DECOMPILED_BINARIES_DATA_DIR, PROGRAM_GRAPH_DATASET_RAW_DIR)

RELOAD_DATASET = False
programs_datamodule, functions_vocabulary, trainer = prepare_default_experiment(reload_dataset=RELOAD_DATASET,
                                                                                dataset_dirs=DatasetDirs())

print("Creating Model...")
EMBEDDING_SIZE = 32
pretrained_model_path = '/home/jonathan/Desktop/Thesis/routinio2.1/dire_neural_model/dire_neural_model/data/data/saved_models/model.hybrid.bin'
model = FunctionNamer(EMBEDDING_SIZE, functions_vocabulary,
                      dire_pre_trained_model_path=pretrained_model_path,
                      train_bottom_model=True,
                      top_neural_model=TopNeuralModels.GCN)
print("Training Model...")
trainer.fit(model, programs_datamodule)
