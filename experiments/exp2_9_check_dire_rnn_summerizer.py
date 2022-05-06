from NeuralModels.function_namer import FunctionNamer, WordGuessers
from NeuralModels.sub_models import Summerizer, TopNeuralModels
from config import nero_small_train_dataset_dirs, EMBEDDING_SIZE, pretrained_dire_model_path, DEBUG_MODE, \
    nero_validation_dataset_dirs

from experiments.experiments_utils import prepare_default_experiment, run_generic_experiment


def run_experiment(seed=44, embedding_size=EMBEDDING_SIZE, learning_rate=0.01):
    RELOAD_DATASET = False
    return run_generic_experiment(
        reload_dataset=RELOAD_DATASET,
        dataset_dirs=nero_small_train_dataset_dirs,
        only_small_dataset=True,
        small_program_size=70,
        name='rnn_siummerizer_double_linear_dire_pretrained',
        seed=seed,
        embedding_size=embedding_size,
        limit_size_for_debug=DEBUG_MODE,
        validation_dataset_dirs=nero_validation_dataset_dirs,
        dire_pre_trained_model_path=pretrained_dire_model_path,
        top_neural_model=TopNeuralModels.ID_DIRE_CHECK,
        word_guesser_type=WordGuessers.DOUBLE_LINEAR_DECODER,
        dire_summerizer=Summerizer.RNN_SUMMERIZER,
        learning_rate=learning_rate,
    )

if __name__ == '__main__':
    run_experiment()
