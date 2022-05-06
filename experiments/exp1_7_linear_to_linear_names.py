from NeuralModels.function_namer import WordGuessers
from NeuralModels.sub_models import BottomNeuralModel, TopNeuralModels

from experiments.experiments_utils import run_generic_experiment
from config import nero_small_train_dataset_dirs, EMBEDDING_SIZE, DEBUG_MODE, \
    nero_validation_dataset_dirs, pretrained_dire_model_path, all_programs_dataset_dirs, \
    all_programs_dataset_dirs_validation


def run_experiment(seed=44, embedding_size=EMBEDDING_SIZE, learning_rate=0.001):
    RELOAD_DATASET = False
    return run_generic_experiment(
        blank_function_namer=True,
        reload_dataset=RELOAD_DATASET,
        # dataset_dirs=nero_small_train_dataset_dirs,
        dataset_dirs=all_programs_dataset_dirs,
        validation_dataset_dirs=all_programs_dataset_dirs_validation,
        only_small_dataset=True,
        small_program_size=200,
        name='id_double_linear',
        seed=seed,
        embedding_size=embedding_size,
        limit_size_for_debug=DEBUG_MODE,
        # validation_dataset_dirs=nero_validation_dataset_dirs,
        dire_pre_trained_model_path=pretrained_dire_model_path,
        bottom_neural_model=BottomNeuralModel.CONSTANT,
        top_neural_model=TopNeuralModels.ID_DIRE_CHECK,
        word_guesser_type=WordGuessers.LINEAR_DECODER,
        # encode_function_as_id=True,
        learning_rate=learning_rate
    )

if __name__ == '__main__':
    run_experiment()