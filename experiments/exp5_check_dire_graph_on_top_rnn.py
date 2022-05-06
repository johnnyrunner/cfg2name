from NeuralModels.function_namer import FunctionNamer, WordGuessers
from NeuralModels.sub_models import TopNeuralModels
from config import DatasetDirs, nero_train_dataset_dirs, nero_small_train_dataset_dirs

from experiments.experiments_utils import prepare_default_experiment

RELOAD_DATASET = False
programs_datamodule, functions_vocabulary, trainer = prepare_default_experiment(reload_dataset=RELOAD_DATASET,
                                                                                dataset_dirs=nero_small_train_dataset_dirs,
                                                                                only_small_dataset=True,
                                                                                small_program_size=70,
                                                                                name='rnn_gcn_dire_pretrained')
def run_experiment(seed=44):
    print("Creating Model...")
    # dire pretrained model location = HYBRID_PRETRAINED_MODEL_PATH
    EMBEDDING_SIZE = 10
    pretrained_model_path = '/home/jonathan/Desktop/Thesis/routinio2.1/dire_neural_model/dire_neural_model/data/data/saved_models/model.hybrid.bin'
    model = FunctionNamer(EMBEDDING_SIZE,
                          functions_vocabulary,
                          dire_pre_trained_model_path=pretrained_model_path,
                          top_neural_model=TopNeuralModels.GCN,
                          word_guesser_type=WordGuessers.RNN_DECODER)
    print("Training Model...")
    trainer.fit(model, programs_datamodule)

if __name__ == '__main__':
    run_experiment()