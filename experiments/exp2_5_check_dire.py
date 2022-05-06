from NeuralModels.function_namer import FunctionNamer, WordGuessers
from NeuralModels.sub_models import TopNeuralModels
from config import nero_small_train_dataset_dirs, EMBEDDING_SIZE, pretrained_dire_model_path, DEBUG_MODE, \
    nero_validation_dataset_dirs

from experiments.experiments_utils import run_generic_experiment


def run_experiment(seed=44, embedding_size=EMBEDDING_SIZE, learning_rate=0.01):
    RELOAD_DATASET = False
    return run_generic_experiment(
        reload_dataset=RELOAD_DATASET,
        dataset_dirs=nero_small_train_dataset_dirs,
        only_small_dataset=True,
        small_program_size=70,
        name='double_linear_dire_pretrained',
        seed=seed,
        embedding_size=embedding_size,
        limit_size_for_debug=DEBUG_MODE,
        validation_dataset_dirs=nero_validation_dataset_dirs,
        dire_pre_trained_model_path=pretrained_dire_model_path,
        top_neural_model=TopNeuralModels.ID_DIRE_CHECK,
        word_guesser_type=WordGuessers.DOUBLE_LINEAR_DECODER,
        learning_rate=learning_rate,
    )

# def run_experiment(seed=44, embedding_size=EMBEDDING_SIZE, learning_rate=0.01):
#     RELOAD_DATASET = True
#     programs_datamodule, functions_vocabulary, trainer = prepare_default_experiment(reload_dataset=RELOAD_DATASET,
#                                                                                     dataset_dirs=nero_small_train_dataset_dirs,
#                                                                                     only_small_dataset=True,
#                                                                                     small_program_size=70,
#                                                                                     name='double_linear_dire_pretrained',
#                                                                                     seed=seed,
#                                                                                     embedding_size=embedding_size,
#                                                                                     limit_size_for_debug=DEBUG_MODE,
#                                                                                     validation_dataset_dirs=nero_validation_dataset_dirs)
#     print("Creating Model...")
#     # dire pretrained model location = HYBRID_PRETRAINED_MODEL_PATH
#     model = FunctionNamer(embedding_size,
#                           functions_vocabulary,
#                           dire_pre_trained_model_path=pretrained_dire_model_path,
#                           top_neural_model=TopNeuralModels.ID_DIRE_CHECK,
#                           word_guesser_type=WordGuessers.DOUBLE_LINEAR_DECODER,
#                           learning_rate=learning_rate,
#                           )
#     print("Training Model...")
#     trainer.fit(model, programs_datamodule)
#     test_list = trainer.test()
#     model_dict = {}
#     model_dict['embedding_size'] = embedding_size
#     model_dict['seed'] = seed
#     model_dict['learning_rate'] = learning_rate
#     test_list.append(model_dict)
#     return test_list


if __name__ == '__main__':
    run_experiment()
