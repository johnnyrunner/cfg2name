from NeuralModels.blank_function_namer import BlankFunctionNamer
from NeuralModels.sub_models import TopNeuralModels

from experiments.experiments_utils import prepare_default_experiment
from config import nero_train_dataset_dirs

RELOAD_DATASET = False
programs_datamodule, functions_vocabulary, trainer = prepare_default_experiment(reload_dataset=RELOAD_DATASET,
                                                                                dataset_dirs=nero_train_dataset_dirs)


print("Creating Model...")
EMBEDDING_SIZE = 10
# dire pretrained model location = HYBRID_PRETRAINED_MODEL_PATH
model = BlankFunctionNamer(EMBEDDING_SIZE, functions_vocabulary, top_neural_model=TopNeuralModels.GCN)
print("Training Model...")
trainer.fit(model, programs_datamodule)
