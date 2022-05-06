from NeuralModels.function_namer import FunctionNamer, WordGuessers
from NeuralModels.sub_models import TopNeuralModels
from config import nero_small_train_dataset_dirs, EMBEDDING_SIZE, DEBUG_MODE, nero_validation_dataset_dirs, \
    pretrained_dire_model_path
from experiments.experiments_utils import run_generic_experiment


def run_experiment(seed=44, embedding_size=EMBEDDING_SIZE, learning_rate=0.01, two_layered=False):
    RELOAD_DATASET = False
    return run_generic_experiment(
        reload_dataset=RELOAD_DATASET,
        dataset_dirs=nero_small_train_dataset_dirs,
        only_small_dataset=True,
        small_program_size=70,
        name='double_linear_gcn_dire_pretrained',
        seed=seed,
        embedding_size=embedding_size,
        limit_size_for_debug=DEBUG_MODE,
        validation_dataset_dirs=nero_validation_dataset_dirs,
        dire_pre_trained_model_path=pretrained_dire_model_path,
        top_neural_model=TopNeuralModels.GCN,
        word_guesser_type=WordGuessers.DOUBLE_LINEAR_DECODER,
        learning_rate=learning_rate,
        two_layered=False,
    )
#
#
# def run_experiment(seed=44, RELOAD_DATASET = False, learning_rate=0.01):
#     programs_datamodule, functions_vocabulary, trainer = prepare_default_experiment(reload_dataset=RELOAD_DATASET,
#                                                                                     dataset_dirs=nero_small_train_dataset_dirs,
#                                                                                     only_small_dataset=True,
#                                                                                     small_program_size=70,
#                                                                                     name='double_linear_gcn_dire_pretrained',
#                                                                                     seed=seed)
#
#     print("Creating Model...")
#     # dire pretrained model location = HYBRID_PRETRAINED_MODEL_PATH
#     EMBEDDING_SIZE = 10
#     pretrained_model_path = '/home/jonathan/Desktop/Thesis/routinio2.1/dire_neural_model/dire_neural_model/data/data/saved_models/model.hybrid.bin'
#     model = FunctionNamer(EMBEDDING_SIZE,
#                           functions_vocabulary,
#                           dire_pre_trained_model_path=pretrained_model_path,
#                           top_neural_model=TopNeuralModels.GCN_ON_DIRE,
#                           word_guesser_type=WordGuessers.DOUBLE_LINEAR_DECODER,
#                           learning_rate=learning_rate,
#
#                           )
#     print("Training Model...")
#     trainer.fit(model, programs_datamodule)

if __name__ == '__main__':
    run_experiment()