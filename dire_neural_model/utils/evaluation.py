import logging
import math
from typing import Dict, List, Any

import numpy as np
import torch

import pandas as pd

import editdistance

import os, sys
from ASR_metrics import utils as metrics
from torch import Tensor

from config import env_vars
from utils.general_utils import lists_are_equal

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
print(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from dire_neural_model.model.model import RenamingModel
from dire_neural_model.utils.dataset import Dataset

EPSILON = 0.00000001


def untensor_if_tensor(a):
    if isinstance(a, Tensor):
        return a.item()
    return a


class Evaluator(object):
    @staticmethod
    def get_soft_metrics(pred_name: str, gold_name: str) -> Dict:
        edit_distance = float(editdistance.eval(pred_name, gold_name))
        cer = float(edit_distance / len(gold_name))
        acc = float(pred_name == gold_name)

        return dict(edit_distance=edit_distance,
                    ref_len=len(gold_name),
                    cer=cer,
                    accuracy=acc)

    @staticmethod
    def calc_prec_recall_f1_scores(should_validate_logits, should_validate_y):
        argmax_should_validate_logits = should_validate_logits.argmax(dim=-1).cpu()
        one_logits = torch.zeros(should_validate_logits.shape).scatter(-1, argmax_should_validate_logits.unsqueeze(-1),
                                                                       1.0)
        one_logits_mean = one_logits.mean(axis=1)  # this is due to irrelevance of order
        one_logits_mean = (one_logits_mean > 0).type(torch.IntTensor)
        if env_vars.use_gpu:
            one_logits_mean = one_logits_mean.cuda()
        number_of_well_predicted_for_each_function = (should_validate_y * one_logits_mean).sum(axis=1)
        precision_per_function = number_of_well_predicted_for_each_function / (
                EPSILON + one_logits_mean.sum(axis=1))
        recall_per_function = number_of_well_predicted_for_each_function / (EPSILON + should_validate_y.sum(axis=1))
        avg_precision, avg_recall = precision_per_function.mean(), recall_per_function.mean()
        f1_score_per_function = 2 * avg_precision * avg_recall / (EPSILON + avg_precision + avg_recall)
        avg_f1_score = f1_score_per_function.mean()
        if math.isnan(avg_f1_score):
            avg_f1_score = 0
        return avg_f1_score, avg_precision, avg_recall, precision_per_function.cpu().numpy(), recall_per_function.cpu().numpy()

    @staticmethod
    def functions_names_evaluation(should_validate_y, should_validate_logits,
                                   function_id_2_word, should_validate_mask, print_names_and_predictions=False,
                                   words_sets_in_train=None):
        if sum(should_validate_mask) > 0:
            avg_f1_score, avg_precision, avg_recall, \
            precision_per_function, recall_per_function = Evaluator.calc_prec_recall_f1_scores(should_validate_logits,
                                                                                               should_validate_y)

            concat_seperated_words_in_predicted, concat_words_in_function, words_id_in_predicted_function_unique = Evaluator.get_words_from_logits_tensor(
                function_id_2_word, should_validate_logits)

            concat_seperated_words_in_y, concat_words_in_y, set_words_in_ys, words_id_in_y_unique = Evaluator.get_words_from_ones_tensor(
                function_id_2_word, should_validate_y)

            cer_per_function = Evaluator.calculate_cer_per_function(concat_words_in_function, concat_words_in_y)

            were_all_words_predicted = Evaluator.calculate_full_Accuracy_per_function(
                words_id_in_predicted_function_unique,
                words_id_in_y_unique)

            if words_sets_in_train is not None:
                is_function_in_train = [set_words_in_y in words_sets_in_train for set_words_in_y in set_words_in_ys]
                words_in_train = set([word for word_set in words_sets_in_train for word in word_set])
                is_function_partly_in_train = [np.any([a in words_in_train for a in set_words_in_y]) for set_words_in_y
                                               in set_words_in_ys]
                is_function_all_words_in_train = [np.all([a in words_in_train for a in set_words_in_y]) for
                                                  set_words_in_y in set_words_in_ys]
                is_function_halfly_in_train = [np.any(
                    [len(set_words_in_y.intersection(word_set_in_train)) > 0.5 * len(set_words_in_y) for
                     word_set_in_train in words_sets_in_train]) for set_words_in_y in set_words_in_ys]
                is_function_only_halfly_in_train = [np.any([0.8 * len(set_words_in_y) > len(
                    set_words_in_y.intersection(word_set_in_train)) > 0.2 * len(set_words_in_y) for word_set_in_train in
                                                            words_sets_in_train]) for set_words_in_y in set_words_in_ys]

                is_function_only_halfly_in_train_not_in_train = [(partly and not in_train) for partly, in_train in
                                                                 zip(is_function_only_halfly_in_train,
                                                                     is_function_in_train)]

            results_dataframe = pd.DataFrame(
                list(zip(concat_seperated_words_in_predicted, concat_seperated_words_in_y, cer_per_function,
                         precision_per_function, recall_per_function, is_function_in_train, were_all_words_predicted,
                         is_function_partly_in_train, is_function_all_words_in_train, is_function_halfly_in_train,
                         is_function_only_halfly_in_train, is_function_only_halfly_in_train_not_in_train)),
                columns=['concatenated_words_in_predicted', 'concatenated_words_in_y', 'cer',
                         'precision', 'recall', 'is_function_in_train', 'were_all_words_predicted',
                         'is_function_partly_in_train', 'is_function_all_words_in_train', 'is_function_halfly_in_train',
                         'is_function_only_halfly_in_train', 'is_function_only_halfly_in_train_not_in_train'])

            results_dataframe['f1_score'] = Evaluator.calc_f1_score_per_series(results_dataframe)
            results_dataframe = results_dataframe.sort_values(by='f1_score', ascending=False)

            if print_names_and_predictions:
                pd.set_option("display.max_rows", None, "display.max_columns", None)
                pd.set_option('display.width', 500)
                print(' ')
                print(results_dataframe)

            percentage_of_exact_word_prediction = sum(were_all_words_predicted) / len(were_all_words_predicted)

            avg_f1_for_in_train = results_dataframe[results_dataframe['is_function_in_train']]['f1_score'].mean()
            if math.isnan(avg_f1_for_in_train):
                avg_f1_for_in_train = 0

            avg_f1_for_not_in_train = results_dataframe[~results_dataframe['is_function_in_train']]['f1_score'].mean()
            if math.isnan(avg_f1_for_not_in_train):
                avg_f1_for_not_in_train = 0

            avg_f1_for_partly_in_train = results_dataframe[results_dataframe['is_function_partly_in_train']][
                'f1_score'].mean()
            if math.isnan(avg_f1_for_partly_in_train):
                avg_f1_for_partly_in_train = 0

            avg_f1_for_all_words_in_train = results_dataframe[results_dataframe['is_function_all_words_in_train']][
                'f1_score'].mean()
            if math.isnan(avg_f1_for_all_words_in_train):
                avg_f1_for_all_words_in_train = 0

            avg_f1_for_halfly_in_train = results_dataframe[results_dataframe['is_function_halfly_in_train']][
                'f1_score'].mean()
            if math.isnan(avg_f1_for_halfly_in_train):
                avg_f1_for_halfly_in_train = 0

            avg_f1_for_only_halfly_in_train = results_dataframe[results_dataframe['is_function_only_halfly_in_train']][
                'f1_score'].mean()
            if math.isnan(avg_f1_for_only_halfly_in_train):
                avg_f1_for_only_halfly_in_train = 0
                ### do not edit names in this dictionary, they are crucial for analysis of the weighted sums they are in.

            avg_f1_for_only_halfly_in_train_not_in_train = \
                results_dataframe[results_dataframe['is_function_only_halfly_in_train_not_in_train']][
                    'f1_score'].mean()
            if math.isnan(avg_f1_for_only_halfly_in_train_not_in_train):
                avg_f1_for_partly_in_train_not_in_train = 0

            number_in_train = sum(is_function_in_train)
            real_batch_size = sum(should_validate_mask)

            results = dict(
                avg_f1_score=(untensor_if_tensor(avg_f1_score), real_batch_size),
                avg_precision_score=(untensor_if_tensor(avg_precision), real_batch_size),
                avg_recall_score=(untensor_if_tensor(avg_recall), real_batch_size),
                avg_cer=(untensor_if_tensor(results_dataframe['cer'].mean()), real_batch_size),
                percentage_of_exact_word_prediction=(
                    untensor_if_tensor(percentage_of_exact_word_prediction), real_batch_size),
                avg_f1_for_in_train=(untensor_if_tensor(avg_f1_for_in_train), number_in_train),
                avg_f1_for_not_in_train=(
                    untensor_if_tensor(avg_f1_for_not_in_train), real_batch_size - number_in_train),
                avg_f1_for_all=(untensor_if_tensor(results_dataframe['f1_score'].mean()), real_batch_size),
                percentage_of_functions_in_train=(untensor_if_tensor(
                    sum(results_dataframe['is_function_in_train']) / len(
                        results_dataframe['is_function_in_train'])), real_batch_size),
                real_batch_size=real_batch_size,
                number_in_train=number_in_train,
                avg_f1_for_partly_in_train=(avg_f1_for_partly_in_train, sum(is_function_partly_in_train)),
                avg_f1_for_all_words_in_train=(avg_f1_for_all_words_in_train, sum(is_function_all_words_in_train)),
                avg_f1_for_halfly_in_train=(avg_f1_for_halfly_in_train, sum(is_function_halfly_in_train)),
                avg_f1_for_only_halfly_in_train=(
                    avg_f1_for_only_halfly_in_train, sum(is_function_only_halfly_in_train)),
                avg_f1_for_partly_in_train_not_in_train=(
                    avg_f1_for_only_halfly_in_train_not_in_train, sum(is_function_only_halfly_in_train_not_in_train))
            )
            return results
        return {}

    @staticmethod
    def calc_f1_score_per_series(results_dataframe):
        return 2 * results_dataframe['precision'] * results_dataframe['recall'] / \
               (EPSILON + results_dataframe['precision'] + results_dataframe['recall'])

    @staticmethod
    def calculate_full_Accuracy_per_function(words_id_in_predicted_function_unique, words_id_in_y_unique):
        were_all_words_predicted = [1 if lists_are_equal(ids_y, ids_predicted) else 0 for (ids_y, ids_predicted) in
                                    zip(words_id_in_y_unique, words_id_in_predicted_function_unique)]
        return were_all_words_predicted

    @staticmethod
    def calculate_cer_per_function(concat_words_in_function, concat_words_in_y):
        cer_per_function = [metrics.calculate_cer(predicted_function_name, function_name) for
                            predicted_function_name, function_name in zip(concat_words_in_function, concat_words_in_y)]
        return cer_per_function

    @staticmethod
    def get_words_from_ones_tensor(function_id_2_word, should_validate_y):
        words_id_in_y = should_validate_y.cpu().numpy().nonzero()
        words_id_in_y_per_function = [[j for (i, j) in zip(words_id_in_y[0], words_id_in_y[1]) if i == index] for index
                                      in range(should_validate_y.shape[0])]
        words_id_in_y_unique = [np.unique(row) for row in words_id_in_y_per_function]
        words_in_y = [[function_id_2_word[i] for i in row] for row in words_id_in_y_unique]
        concat_words_in_y = [''.join(words_in_function) for words_in_function in words_in_y]
        concat_seperated_words_in_y = ['-'.join(words_in_function) for words_in_function in words_in_y]
        set_words_in_ys = [frozenset(words_in_function) for words_in_function in words_in_y]
        return concat_seperated_words_in_y, concat_words_in_y, set_words_in_ys, words_id_in_y_unique

    @staticmethod
    def get_words_from_logits_tensor(function_id_2_word, should_validate_logits):
        words_id_in_predicted_function = should_validate_logits.argmax(dim=-1).contiguous().cpu().numpy()
        words_id_in_predicted_function_unique = [np.unique(row) for row in words_id_in_predicted_function]
        predicted_words_in_functions = [[function_id_2_word[i] for i in row] for row in
                                        words_id_in_predicted_function_unique]
        concat_words_in_function = [''.join(words_in_function) for words_in_function in predicted_words_in_functions]
        concat_seperated_words_in_predicted = ['-'.join(words_in_function) for words_in_function in
                                               predicted_words_in_functions]
        return concat_seperated_words_in_predicted, concat_words_in_function, words_id_in_predicted_function_unique

    @staticmethod
    def average(metrics_list: List[Dict]) -> Dict:
        agg_results = dict()
        for metrics in metrics_list:
            for key, val in metrics.items():
                agg_results.setdefault(key, []).append(val)

        avg_results = dict()
        avg_results['corpus_cer'] = sum(agg_results['edit_distance']) / sum(agg_results['ref_len'])

        for key, val in agg_results.items():
            avg_results[key] = np.average(val)

        return avg_results

    @staticmethod
    def evaluate_ppl(model: RenamingModel, dataset: Dataset, config: Dict, predicate: Any = None):
        if predicate is None:
            predicate = lambda e: True

        eval_batch_size = config['train']['batch_size']
        data_iter = dataset.batch_iterator(batch_size=eval_batch_size,
                                           train=False, progress=True,
                                           return_examples=False,
                                           return_prediction_target=True,
                                           config=model.config,
                                           num_readers=config['train']['num_readers'],
                                           num_batchers=config['train']['num_batchers'])

        was_training = model.training
        model.eval()
        cum_log_probs = 0.
        cum_num_examples = 0
        with torch.no_grad():
            for batch in data_iter:
                from dire_neural_model.utils import nn_util
                nn_util.to(batch.tensor_dict, model.device)
                result = model(batch.tensor_dict, batch.tensor_dict['prediction_target'])
                log_probs = result['batch_log_prob'].cpu().tolist()
                for e_id, test_meta in enumerate(batch.tensor_dict['test_meta']):
                    if predicate(test_meta):
                        log_prob = log_probs[e_id]
                        cum_log_probs += log_prob
                        cum_num_examples += 1

        ppl = np.exp(-cum_log_probs / cum_num_examples)

        if was_training:
            model.train()

        return ppl

    @staticmethod
    def evaluate_examples_batch(examples, rename_results):
        # a function for kavitzky's FunctionNamer
        example_acc_list = []
        variable_acc_list = []
        all_old_names = examples['_variable_name_map'][0]['all_old_names']
        all_gold_new_names = examples['_variable_name_map'][0]['all_gold_new_names']

        all_preds = []
        for old_names, gold_new_names, rename_result in zip(all_old_names, all_gold_new_names, rename_results):
            preds = []
            example_pred_accs = []
            top_rename_result = rename_result[0]
            for old_name, gold_new_name in zip(old_names, gold_new_names):
                try:
                    pred = top_rename_result[old_name]
                    pred_new_name = pred['new_name']
                    preds.append(pred_new_name)
                    var_metric = Evaluator.get_soft_metrics(pred_new_name, gold_new_name)
                    # is_correct = pred_new_name == gold_new_name
                    example_pred_accs.append(var_metric)
                except:
                    print('bad function name')
            all_preds.append(preds)
            variable_acc_list.extend(example_pred_accs)
            example_acc_list.append(example_pred_accs)

        # print(all_old_names)
        # print(all_gold_new_names)
        # print(all_preds)
        num_variables = len(variable_acc_list)
        corpus_acc = Evaluator.average(variable_acc_list)
        # print(f'num_variables: {num_variables}')
        # print(f'corpus_acc: {corpus_acc}')

        return num_variables, corpus_acc

    @staticmethod
    def decode_and_evaluate(model: RenamingModel, dataset: Dataset, config: Dict, return_results=False,
                            eval_batch_size=None):
        if eval_batch_size is None:
            eval_batch_size = config['train']['eval_batch_size'] if 'eval_batch_size' in config['train'] else \
                config['train']['batch_size']
        data_iter = dataset.batch_iterator(batch_size=eval_batch_size,
                                           train=False, progress=True,
                                           return_examples=True,
                                           config=model.config,
                                           num_readers=config['train']['num_readers'],
                                           num_batchers=config['train']['num_batchers'])

        was_training = model.training
        model.eval()
        example_acc_list = []
        variable_acc_list = []
        need_rename_cases = []

        func_name_in_train_acc_list = []
        func_name_not_in_train_acc_list = []
        func_body_in_train_acc_list = []
        func_body_not_in_train_acc_list = []

        all_examples = dict()

        with torch.no_grad():
            for batch in data_iter:
                examples = batch.examples
                rename_results = model.predict(examples)
                for example, rename_result in zip(examples, rename_results):
                    example_pred_accs = []

                    top_rename_result = rename_result[0]
                    for old_name, gold_new_name in example._variable_name_map.items():
                        pred = top_rename_result[old_name]
                        pred_new_name = pred['new_name']
                        var_metric = Evaluator.get_soft_metrics(pred_new_name, gold_new_name)
                        # is_correct = pred_new_name == gold_new_name
                        example_pred_accs.append(var_metric)

                        if gold_new_name != old_name:  # and gold_new_name in model.vocab.target:
                            need_rename_cases.append(var_metric)

                            if example.test_meta['function_name_in_train']:
                                func_name_in_train_acc_list.append(var_metric)
                            else:
                                func_name_not_in_train_acc_list.append(var_metric)

                            if example.test_meta['function_body_in_train']:
                                func_body_in_train_acc_list.append(var_metric)
                            else:
                                func_body_not_in_train_acc_list.append(var_metric)

                    variable_acc_list.extend(example_pred_accs)
                    example_acc_list.append(example_pred_accs)

                    if return_results:
                        all_examples[
                            example.binary_file['file_name'] + '_' + str(example.binary_file['line_num'])] = (
                            rename_result, Evaluator.average(example_pred_accs))
                        # all_examples.append((example, rename_result, example_pred_accs))

        valid_example_num = len(example_acc_list)
        num_variables = len(variable_acc_list)
        corpus_acc = Evaluator.average(variable_acc_list)

        if was_training:
            model.train()

        eval_results = dict(corpus_acc=corpus_acc,
                            corpus_need_rename_acc=Evaluator.average(need_rename_cases),
                            func_name_in_train_acc=Evaluator.average(func_name_in_train_acc_list),
                            func_name_not_in_train_acc=Evaluator.average(func_name_not_in_train_acc_list),
                            func_body_in_train_acc=Evaluator.average(func_body_in_train_acc_list),
                            func_body_not_in_train_acc=Evaluator.average(func_body_not_in_train_acc_list),
                            num_variables=num_variables,
                            num_valid_examples=valid_example_num)

        if return_results:
            return eval_results, all_examples
        return eval_results

    if __name__ == '__main__':
        print(Evaluator.get_soft_metrics('file_name', 'filename'))
