from NeuralModels.function_namer import WordGuessers
from NeuralModels.sub_models import BottomNeuralModel, TopNeuralModels
from config import nero_small_train_dataset_dirs, EMBEDDING_SIZE, DEBUG_MODE, nero_validation_dataset_dirs, \
    pretrained_dire_model_path
from experiments.experiments_utils import run_generic_experiment

def run_experiment(seed=44, embedding_size=100, learning_rate=0.001):
    RELOAD_DATASET = False
    return run_generic_experiment(
        reload_dataset=RELOAD_DATASET,
        dataset_dirs=nero_small_train_dataset_dirs,
        only_small_dataset=True,
        small_program_size=70,
        name='give_answers_gat_linear', # TODO: check this is really gets very high train accuracy!
        seed=seed,
        embedding_size=embedding_size,
        limit_size_for_debug=DEBUG_MODE,
        validation_dataset_dirs=nero_validation_dataset_dirs,
        dire_pre_trained_model_path=pretrained_dire_model_path,
        bottom_neural_model=BottomNeuralModel.NAMES,
        top_neural_model=TopNeuralModels.GAT,
        word_guesser_type=WordGuessers.LINEAR_DECODER,
        learning_rate=learning_rate,
        give_answers=True,
        max_epochs=2000,

    )

if __name__ == '__main__':
    run_experiment()