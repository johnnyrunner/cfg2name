from NeuralModels.function_namer import WordGuessers
from NeuralModels.sub_models import BottomNeuralModel, TopNeuralModels
from config import nero_small_train_dataset_dirs, EMBEDDING_SIZE, DEBUG_MODE, nero_validation_dataset_dirs, \
    pretrained_dire_model_path, test_edges_dataset_dirs
from experiments.experiments_utils import run_generic_experiment

def run_experiment(seed=44, embedding_size=EMBEDDING_SIZE, learning_rate=0.0005):
    RELOAD_DATASET = True
    return run_generic_experiment(
        reload_dataset=RELOAD_DATASET,
        dataset_dirs=test_edges_dataset_dirs,
        only_small_dataset=True,
        small_program_size=1000,
        name='names_gat_linear_name',
        seed=seed,
        embedding_size=embedding_size,
        limit_size_for_debug=DEBUG_MODE,
        validation_dataset_dirs=test_edges_dataset_dirs,
        dire_pre_trained_model_path=pretrained_dire_model_path,
        bottom_neural_model=BottomNeuralModel.NAMES,
        top_neural_model=TopNeuralModels.GAT,
        word_guesser_type=WordGuessers.LINEAR_DECODER,
        learning_rate=learning_rate,
        portion_not_real=0.1,
        max_epochs=2000,
    )

if __name__ == '__main__':
    run_experiment()