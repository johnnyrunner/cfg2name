from NeuralModels.function_namer import FunctionNamer, WordGuessers
from NeuralModels.sub_models import TopNeuralModels
from config import nero_small_train_dataset_dirs, EMBEDDING_SIZE, pretrained_dire_model_path

from experiments.experiments_utils import prepare_default_experiment


def run_experiment(seed=44, embedding_size=EMBEDDING_SIZE, learning_rate=0.01, reload_dataset=False):
    programs_datamodule, functions_vocabulary, trainer = prepare_default_experiment(reload_dataset=reload_dataset,
                                                                                    dataset_dirs=nero_small_train_dataset_dirs,
                                                                                    only_small_dataset=True,
                                                                                    small_program_size=70,
                                                                                    name='triple_linear_dire_pretrained',
                                                                                    seed=seed)

    print("Creating Model...")
    # dire pretrained model location = HYBRID_PRETRAINED_MODEL_PATH
    model = FunctionNamer(embedding_size,
                          functions_vocabulary,
                          dire_pre_trained_model_path=pretrained_dire_model_path,
                          top_neural_model=TopNeuralModels.ID_DIRE_CHECK,
                          word_guesser_type=WordGuessers.TRIPLE_LINEAR_DECODER,
                          learning_rate=learning_rate,
                          )
    print("Training Model...")
    trainer.fit(model, programs_datamodule)


if __name__ == '__main__':
    run_experiment()
