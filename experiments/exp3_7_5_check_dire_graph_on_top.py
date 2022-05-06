from NeuralModels.function_namer import WordGuessers
from NeuralModels.sub_models import TopNeuralModels
from config import nero_small_train_dataset_dirs, EMBEDDING_SIZE, nero_validation_dataset_dirs, \
    pretrained_dire_model_path, DEBUG_MODE, nero_test_dataset_dirs, all_programs_dataset_dirs, \
    all_programs_dataset_dirs_validation
from experiments.experiments_utils import run_generic_experiment

def run_experiment(seed=45, embedding_size=EMBEDDING_SIZE, learning_rate=0.001, two_layered=False):
    RELOAD_DATASET = False
    return run_generic_experiment(
        reload_dataset=RELOAD_DATASET,
        dataset_dirs=all_programs_dataset_dirs,
        validation_dataset_dirs=all_programs_dataset_dirs_validation,
        only_small_dataset=True,
        small_program_size=200,
        name='linear_gat_dire_pretrained',
        seed=seed,
        embedding_size=embedding_size,
        limit_size_for_debug=DEBUG_MODE,
        # limit_size_for_debug=True,
        # validation_dataset_dirs=nero_validation_dataset_dirs,
        dire_pre_trained_model_path=pretrained_dire_model_path,
        top_neural_model=TopNeuralModels.GAT,
        word_guesser_type=WordGuessers.LINEAR_DECODER,
        learning_rate=learning_rate,
        two_layered=two_layered
    )


if __name__ == '__main__':
    print(run_experiment())

