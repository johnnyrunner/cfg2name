from NeuralModels.function_namer import FunctionNamer, WordGuessers
from NeuralModels.sub_models import TopNeuralModels
from config import DatasetDirs, nero_small_train_dataset_dirs

from experiments.experiments_utils import prepare_default_experiment


RELOAD_DATASET = False
programs_datamodule, functions_vocabulary, trainer = prepare_default_experiment(reload_dataset=RELOAD_DATASET,
                                                                                dataset_dirs=nero_small_train_dataset_dirs,
                                                                                only_small_dataset=True,
                                                                                small_program_size=70,
                                                                                name='id_linear_dire_pretrained')
print("Creating Model...")
# dire pretrained model location = HYBRID_PRETRAINED_MODEL_PATH
EMBEDDING_SIZE = 32
pretrained_model_path = '/home/jonathan/Desktop/Thesis/routinio2.1/dire_neural_model/dire_neural_model/data/data/saved_models/model.hybrid.bin'
model = FunctionNamer(EMBEDDING_SIZE, functions_vocabulary,
                      dire_pre_trained_model_path=pretrained_model_path,
                      top_neural_model=TopNeuralModels.ID_DIRE_CHECK,
                      word_guesser_type=WordGuessers.LINEAR_DECODER,
                      learning_rate=0.01,
                      )
print("Training Model...")
trainer.fit(model, programs_datamodule)
