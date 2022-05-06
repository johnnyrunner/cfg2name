import copy

import torch
from pytorch_lightning import Trainer, loggers

from Datasets import dire_cache
from Datasets.programs_dataset import ProgramsDataset, SMALL_PROGRAM_SIZE
from NeuralModels.blank_function_namer import BlankFunctionNamer
from NeuralModels.function_namer import FunctionNamer
from NeuralModels.programs_data_module import ProgramsDataModule
from NeuralModels.sub_models import BottomNeuralModel

from config import DatasetDirs, RUN_START_DATE, STRIPPED, env_vars, ORIGINAL, LOGGING_BASE_DIR
from utils.files_utils import delete_dataset_if_needed
from utils.general_utils import set_seed
from utils.graph_utils import MergeFunctionsDataIntoNetworkx


def prepare_default_experiment(reload_dataset, dataset_dirs: DatasetDirs, only_small_dataset=False, seed=44,
                               name='default', small_program_size=SMALL_PROGRAM_SIZE, limit_size_for_debug=False,
                               max_epochs=250, embedding_size=None, stripped_or_original=ORIGINAL,
                               validation_dataset_dirs: DatasetDirs = None, test_dataset_dirs: DatasetDirs = None,
                               size_of_subsample=50, logging_base=LOGGING_BASE_DIR, functions_vocab_size=2000, *args,
                               **kwargs):
    inputs = locals()
    all_inputs = {**inputs, **kwargs}
    all_inputs['kwargs'] = 'kwargs'
    delete_dataset_if_needed(reload_dataset=reload_dataset, dataset_root_dir=dataset_dirs.root_dir)
    if validation_dataset_dirs is not None:
        delete_dataset_if_needed(reload_dataset=reload_dataset, dataset_root_dir=validation_dataset_dirs.root_dir)
    if reload_dataset:
        MergeFunctionsDataIntoNetworkx(dataset_dirs.decompiled_binaries_data_dir, dataset_dirs.raw_dir)
        if validation_dataset_dirs is not None:
            print('validation merge')
            MergeFunctionsDataIntoNetworkx(validation_dataset_dirs.decompiled_binaries_data_dir,
                                           validation_dataset_dirs.raw_dir)
    set_seed(seed)
    print("Building ProgramsDataModule...")
    programs_dataset = ProgramsDataset(dataset_dirs.root_dir, load_dire_vocab_from_file=True,
                                       stripped_or_original=stripped_or_original, only_small_dataset=only_small_dataset,
                                       small_program_size=small_program_size, size_of_subsamples=size_of_subsample,
                                       functions_vocab_size=functions_vocab_size,  # retrain_functions_vocab=True,
                                       )
    if validation_dataset_dirs is not None:
        validation_dataset = ProgramsDataset(validation_dataset_dirs.root_dir, load_dire_vocab_from_file=True,
                                             stripped_or_original=stripped_or_original,
                                             only_small_dataset=only_small_dataset,
                                             small_program_size=small_program_size,
                                             load_vocab_from_dataset=programs_dataset,
                                             size_of_subsamples=size_of_subsample,
                                             )
    else:
        validation_dataset = None

    if test_dataset_dirs is not None:
        test_dataset = ProgramsDataset(test_dataset_dirs.root_dir, load_dire_vocab_from_file=True,
                                       stripped_or_original=stripped_or_original,
                                       only_small_dataset=only_small_dataset,
                                       small_program_size=small_program_size,
                                       load_vocab_from_dataset=programs_dataset,
                                       size_of_subsamples=size_of_subsample,
                                       )
    else:
        test_dataset = None
    programs_datamodule = ProgramsDataModule((programs_dataset), 70, 20, batch_size=1,
                                             validation_dataset=(validation_dataset),
                                             test_dataset=(test_dataset))

    functions_vocabulary = programs_dataset.functions_vocab

    if limit_size_for_debug:
        tb_logger = loggers.TensorBoardLogger(save_dir=logging_base + '/dbg_lightning_logs/' + RUN_START_DATE,
                                              name=name)
    else:
        tb_logger = loggers.TensorBoardLogger(save_dir=logging_base + '/lightning_logs/' + RUN_START_DATE,
                                              name=name + '_' + str(embedding_size))

    if env_vars.use_gpu:
        if limit_size_for_debug:
            trainer = Trainer(gpus=1, logger=tb_logger, max_epochs=2, limit_train_batches=2, limit_val_batches=2,
                              limit_test_batches=3)
        else:
            trainer = Trainer(gpus=1, logger=tb_logger, max_epochs=max_epochs)
    else:
        if limit_size_for_debug:
            trainer = Trainer(logger=tb_logger, max_epochs=2, limit_train_batches=2, limit_val_batches=2,
                              limit_test_batches=3)
        else:
            trainer = Trainer(logger=tb_logger, max_epochs=max_epochs)
    # TODO: In train should be re-written for really in train after splitting.
    return programs_datamodule, functions_vocabulary, trainer, programs_dataset.all_function_names, all_inputs


def fill_dire_cache(functions_vocabulary, all_function_names, programs_datamodule, *args, **kwargs):
    kwargs['name'] = 'cache_filler'
    old_USE_GPU = copy.deepcopy(env_vars.use_gpu)
    env_vars.change_use_gpu(False)
    _, _, trainer, _, _ = prepare_default_experiment(*args, **kwargs, max_epochs=1)
    model = FunctionNamer(functions_vocabulary=functions_vocabulary, all_train_function_names=all_function_names, *args,
                          **kwargs)

    trainer.fit(model, programs_datamodule)
    print('finished filling')
    env_vars.change_use_gpu(old_USE_GPU)


def get_relevant_logged_metrics(logged_metrics):
    return {k: v for k, v in logged_metrics.items() if '/' not in k}


def run_generic_experiment(bottom_neural_model=BottomNeuralModel.DIRE, blank_function_namer=False, *args, **kwargs):
    programs_datamodule, functions_vocabulary, trainer, all_train_function_names, inputs = prepare_default_experiment(
        bottom_neural_model=bottom_neural_model,
        *args,
        **kwargs)
    print("Creating Model...")
    # dire pretrained model location = HYBRID_PRETRAINED_MODEL_PATH
    print('a')
    if blank_function_namer:
        print('c')

        model = BlankFunctionNamer(functions_vocabulary=functions_vocabulary,
                                   all_train_function_names=all_train_function_names,
                                   bottom_neural_model=bottom_neural_model,
                                   *args, **kwargs)
        print('d')

    else:
        print('e')

        if bottom_neural_model == BottomNeuralModel.DIRE:
            fill_dire_cache(copy.deepcopy(functions_vocabulary),
                            copy.deepcopy(all_train_function_names),
                            # copy.deepcopy(programs_datamodule),
                            programs_datamodule,
                            *args, **kwargs)

        print('creating model')
        model = FunctionNamer(functions_vocabulary=functions_vocabulary, bottom_neural_model=bottom_neural_model,
                              all_train_function_names=all_train_function_names, *args, **kwargs)
    print('b')
    print(torch.cuda.memory_summary())

    print("Training Model...")
    trainer.fit(model, programs_datamodule)
    # test_results_dict = trainer.test()[0]
    base_dict = {}
    base_dict.update(inputs)
    base_dict.update(get_relevant_logged_metrics(trainer.logged_metrics))
    keys_to_remove = ['kwargs', 'args', 'logging_base', 'test_dataset_dirs', 'validation_dataset_dirs',
                      'dataset_dirs', ]
    base_dict = {k: v for k, v in base_dict.items() if k not in keys_to_remove}
    # test_results_dict.update(model.__dict__)
    return base_dict
