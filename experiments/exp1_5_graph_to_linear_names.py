from NeuralModels.blank_function_namer import BlankFunctionNamer
from NeuralModels.function_namer import WordGuessers
from NeuralModels.sub_models import BottomNeuralModel, TopNeuralModels

from experiments.experiments_utils import prepare_default_experiment, get_relevant_logged_metrics
from config import all_programs_dataset_dirs, all_programs_dataset_dirs_validation

RELOAD_DATASET = False
programs_datamodule, functions_vocabulary, trainer, all_train_function_names, inputs = prepare_default_experiment(
    reload_dataset=False,
    dataset_dirs=all_programs_dataset_dirs,
    validation_dataset_dirs=all_programs_dataset_dirs_validation,
    name='constant_gcn_linear')

print("Creating Model...")
EMBEDDING_SIZE = 500
# dire pretrained model location = HYBRID_PRETRAINED_MODEL_PATH
model = BlankFunctionNamer(EMBEDDING_SIZE, functions_vocabulary,
                           bottom_neural_model=BottomNeuralModel.CONSTANT,
                           top_neural_model=TopNeuralModels.GCN,
                           word_guesser_type=WordGuessers.LINEAR_DECODER,
                           learning_rate=0.001,
                           all_function_names=all_train_function_names
                           )
print("Training Model...")
trainer.fit(model, programs_datamodule)

trainer.fit(model, programs_datamodule)
# test_results_dict = trainer.test()[0]
base_dict = {}
base_dict.update(inputs)
base_dict.update(get_relevant_logged_metrics(trainer.logged_metrics))
keys_to_remove = ['kwargs', 'args', 'logging_base', 'test_dataset_dirs', 'validation_dataset_dirs',
                  'dataset_dirs', ]
base_dict = {k: v for k, v in base_dict.items() if k not in keys_to_remove}
# test_results_dict.update(model.__dict__)
print(base_dict)
