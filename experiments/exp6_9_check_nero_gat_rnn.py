from NeuralModels.function_namer import FunctionNamer, WordGuessers
from NeuralModels.sub_models import BottomNeuralModel, TopNeuralModels
from config import nero_small_train_dataset_dirs, EMBEDDING_SIZE, pretrained_dire_model_path, DEBUG_MODE, \
    nero_validation_dataset_dirs

from experiments.experiments_utils import run_generic_experiment


def run_experiment(seed=44, embedding_size=EMBEDDING_SIZE, learning_rate=0.05, two_layered=False):
    RELOAD_DATASET = False
    return run_generic_experiment(
        reload_dataset=RELOAD_DATASET,
        dataset_dirs=nero_small_train_dataset_dirs,
        only_small_dataset=True,
        small_program_size=180,
        name='rnn_GAT_nero_pretrained',
        seed=seed,
        embedding_size=embedding_size,
        limit_size_for_debug=DEBUG_MODE,
        validation_dataset_dirs=nero_validation_dataset_dirs,
        dire_pre_trained_model_path=pretrained_dire_model_path,
        bottom_neural_model=BottomNeuralModel.NERO,
        top_neural_model=TopNeuralModels.GAT,
        word_guesser_type=WordGuessers.RNN_DECODER,
        learning_rate=learning_rate,
        two_layered=two_layered
    )


if __name__ == '__main__':
    run_experiment()
