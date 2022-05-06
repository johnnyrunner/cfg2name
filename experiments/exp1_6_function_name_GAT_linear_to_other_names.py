from NeuralModels.function_namer import WordGuessers
from NeuralModels.sub_models import BottomNeuralModel, TopNeuralModels
from config import nero_small_train_dataset_dirs, EMBEDDING_SIZE, DEBUG_MODE, nero_validation_dataset_dirs, \
    pretrained_dire_model_path, all_programs_dataset_dirs, all_programs_dataset_dirs_validation
from experiments.experiments_utils import run_generic_experiment

def run_experiment(seed=46, embedding_size=500, learning_rate=0.001):
    RELOAD_DATASET = False
    return run_generic_experiment(
        reload_dataset=RELOAD_DATASET,
        dataset_dirs=all_programs_dataset_dirs,
        only_small_dataset=True,
        small_program_size=200,
        name='names_gat_linear_name',
        seed=seed,
        embedding_size=embedding_size,
        limit_size_for_debug=DEBUG_MODE,
        validation_dataset_dirs=all_programs_dataset_dirs_validation,
        dire_pre_trained_model_path=pretrained_dire_model_path,
        bottom_neural_model=BottomNeuralModel.NAMES,
        top_neural_model=TopNeuralModels.GAT,
        word_guesser_type=WordGuessers.LINEAR_DECODER,
        learning_rate=learning_rate,
        portion_not_real=0.9,
        max_epochs=2000,
    )


if __name__ == '__main__':
    run_experiment()