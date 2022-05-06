import math
from unittest import TestCase

import torch

from config import env_vars
from dire_neural_model.utils.evaluation import Evaluator


class TestEvaluator(TestCase):
    def test_functions_names_evaluation(self):
        env_vars.change_use_gpu(False)
        should_validate_y = torch.zeros((3, 4))  # batch_size, number_of_words_in_vocab
        should_validate_y[1, 1] = 1  # b - g.t. function 1 bc
        should_validate_y[1, 2] = 1  # c - g.t. function 1
        should_validate_y[0, 1] = 1  # b - g.t. function 0 ab
        should_validate_y[0, 0] = 1  # a - g.t. function 0
        should_validate_y[2, 3] = 1  # a - g.t. function 3 a

        should_validate_logits = torch.zeros((3, 2, 4))  # batch_size, guessed words, number_of_words_in_vocab
        should_validate_logits[1, 0, 1] = 1  # b - guesses function 1 bc
        should_validate_logits[1, 1, 2] = 1  # c - guesses function 1
        should_validate_logits[0, 0, 2] = 1  # c - guesses function 0 ca
        should_validate_logits[0, 1, 0] = 1  # a - guesses function 0
        should_validate_logits[2, 0, 2] = 1  # a - guesses function 2 c

        functions_id2word = {0: 'a', 1: 'b', 2: 'c', 3: 'd'}
        should_validate_mask = torch.ones((3, 1))
        words_sets_in_train = set([frozenset(['b', 'c']), frozenset(['b', 'a'])])
        print_names_and_predictions = False
        results = Evaluator.functions_names_evaluation(should_validate_y, should_validate_logits, functions_id2word,
                                                       should_validate_mask, print_names_and_predictions,
                                                       words_sets_in_train, )
        avg_precision = (1 + 0.5 + 0) / 3
        avg_recall = (1 + 0.5 + 0) / 3
        avg_f1_scores = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)
        avg_f1_score = (2 * 1 / (1 + 1) + 2 * 0.5 * 0.5 / (0.5 + 0.5) + (2 * 0 * 0 / 0.0001))/3
        percentage_of_exact = 1/3
        avg_f1_for_in_train = (1 + 0.5)/2
        avg_f1_for_not_in_train = 0
        percentage_of_functions_in_train = 2/3
        avg_cer = (0 + (1)/(1+1) + (1)/(1))/3
        print(results)
        assert avg_recall == results['avg_recall_score']
        assert avg_f1_scores == results['avg_f1_score'] == avg_f1_score
        assert percentage_of_exact == results['percentage_of_exact_word_prediction']
        assert math.fabs(avg_f1_for_in_train - results['avg_f1_for_in_train']) < 0.000001
        assert avg_f1_for_not_in_train == results['avg_f1_for_not_in_train']
        assert percentage_of_functions_in_train == results['percentage_of_functions_in_train']
        assert avg_cer == results['avg_cer']
