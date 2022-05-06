import json
from enum import Enum

import pytorch_lightning as pl
from torch import Tensor
from torch.nn import functional as F

from Datasets.dire_cache import dire_cache
from Datasets.nero_cache import nero_cache
from NeuralModels.sub_models import BottomNeuralModel, Id, RnnSummerizer, Summerizer, GCN, GAT_model, TopNeuralModels, \
    RnnDecoderWordsGuesser, LinearDecoderWordsGuesser, bottom_to_output_size
from dire_neural_model.model.model import RenamingModel, ConstantModel, NamesModel
from config import *
from dire_neural_model.utils.evaluation import Evaluator
from utils.graph_utils import plot_graph_data
from vocabulary_builder.functions_vocab import FunctionsVocab
import numpy as np
import torch.nn as nn


def weighted_tuples_mean(values_nums):
    number_of_examples = np.sum([num for num in values_nums['num']])
    if number_of_examples == 0:
        return 0
    weighted_sum = np.sum([num * val for (val, num) in zip(values_nums['val'], values_nums['num'])])
    return weighted_sum / number_of_examples


class WordGuessers(Enum):
    RNN_DECODER = 'rnn_decoder'
    GNEREIC_RNN_DECODER = 'generic_rnn_decoder'
    LINEAR_DECODER = 'linear_decoder'
    DOUBLE_LINEAR_DECODER = 'double_linear_decoder'
    TRIPLE_LINEAR_DECODER = 'triple_linear_decoder'


def get_relevant_nero_vectors_and_graph(nero_ids, graph_data):
    relevant_nero_ids = [nero_id for nero_id in nero_ids if nero_id in nero_cache.examples.keys()]
    nero_relevant_function_names = [nero_id.split('@')[0].replace('*', '_') for nero_id in relevant_nero_ids]
    name_to_nero_id = {name: id_ for (name, id_) in zip(nero_relevant_function_names, relevant_nero_ids)}
    mangled_function_names = graph_data['mangled_function_names'][0]
    name_to_graph_id = {name: id_ for (id_, name) in enumerate(mangled_function_names)}
    # node_ids = graph_data['id'][0]
    functions_in_nero = [name_to_graph_id[function_name] for function_name in mangled_function_names if
                         function_name in nero_relevant_function_names]
    functions_not_in_nero = [name_to_graph_id[function_name] for function_name in mangled_function_names if
                             function_name not in nero_relevant_function_names]

    old_number_of_nodes = sum(graph_data["should_validate"])
    graph_data['should_validate'] = Tensor(
        [sv and (name_to_graph_id[name] not in functions_not_in_nero) for (sv, name) in
         zip(graph_data['should_validate'].cpu().numpy(), mangled_function_names)]).to(device=env_vars.torch_device)
    print(f'original number of nodes {old_number_of_nodes}, new number of nodes {sum(graph_data["should_validate"])}')
    assert len(relevant_nero_ids) >= sum(graph_data['should_validate'])
    edges = [edge for edge in zip(graph_data['edge_index'][0], graph_data['edge_index'][1]) if
             int(edge[0]) not in functions_not_in_nero and int(edge[1]) not in functions_not_in_nero]
    print(f'original number of edges {len(graph_data["edge_index"][0])}, new number of edges {len(edges)}')
    graph_data['edges_index'] = Tensor([[i for i, _ in edges], [j for _, j in edges]]).to(device=env_vars.torch_device)
    nero_vectors = Tensor([nero_cache.examples[
                               name_to_nero_id[nero_name]] if nero_name in nero_relevant_function_names else np.zeros(
        (512,)) for nero_name in mangled_function_names])
    assert set([i for i, val in enumerate(nero_vectors.sum(axis=1)) if val]) == set(functions_in_nero)
    return nero_vectors.to(device=env_vars.torch_device), graph_data


class FunctionNamer(pl.LightningModule):
    def __init__(self, embedding_size,
                 functions_vocabulary: FunctionsVocab,
                 all_train_function_names=None,
                 dire_pre_trained_model_path=None,
                 bottom_neural_model=BottomNeuralModel.DIRE,
                 top_neural_model=TopNeuralModels.ID_DIRE_CHECK,
                 word_guesser_type=WordGuessers.RNN_DECODER,
                 dire_summerizer=Summerizer.ID,
                 dire_summerizer_hidden_size=20,
                 dire_summerizer_num_layers=1,
                 train_bottom_model=False,
                 learning_rate=0.001,
                 give_answers=False,
                 number_not_real=1,
                 portion_not_real=None,
                 *args, **kwargs):
        print(bottom_neural_model)
        print(all_train_function_names)
        assert top_neural_model in TopNeuralModels
        assert word_guesser_type in WordGuessers
        assert bottom_neural_model in BottomNeuralModel
        print(1)
        self.bottom_neural_model = bottom_neural_model
        self.train_bottom_model = train_bottom_model
        self.learning_rate = learning_rate
        super(FunctionNamer, self).__init__()
        self.functions_vocabulary = functions_vocabulary
        self.functions_id2word = self.functions_vocabulary.get_id2word()
        self.words_sets_in_train = set(all_train_function_names)
        self.criterion = nn.BCEWithLogitsLoss()
        dire_config = json.load(open(DIRE_DEFAULT_CONFIG_HYBRID_DIR, 'r'))
        print('f')

        if dire_summerizer == Summerizer.ID:
            self.summerizer = Id(self.bottom_neural_model)

        elif dire_summerizer == Summerizer.RNN_SUMMERIZER:
            self.summerizer = RnnSummerizer(self.bottom_neural_model,
                                            hidden_size=dire_summerizer_hidden_size,
                                            num_layers=dire_summerizer_num_layers)
        if bottom_neural_model == BottomNeuralModel.DIRE:
            self.pre_trained_model_path = dire_pre_trained_model_path
            if self.pre_trained_model_path is not None:
                self.bottom_model = RenamingModel.load(dire_pre_trained_model_path, use_cuda=env_vars.use_gpu,
                                                       new_config=dire_config, should_decode=True)
            else:
                self.bottom_model = RenamingModel.build(dire_config)
        elif bottom_neural_model == BottomNeuralModel.CONSTANT:
            self.bottom_model = ConstantModel()
        elif bottom_neural_model == BottomNeuralModel.NAMES:
            self.bottom_model = NamesModel(number_not_real=number_not_real, give_answers=give_answers,
                                           portion_not_real=portion_not_real)
        print('g')

        self.linear_embedding_size = embedding_size
        if top_neural_model == TopNeuralModels.GCN:
            self.embedding_size = embedding_size
            self.top_model = GCN(self.bottom_neural_model, self.embedding_size,
                                 summerizer=self.summerizer, **kwargs)  # geometric GCN
        if top_neural_model == TopNeuralModels.GAT:
            self.embedding_size = embedding_size
            self.top_model = GAT_model(self.bottom_neural_model, self.embedding_size,
                                       summerizer=self.summerizer, **kwargs)  # geometric GAT
        else:
            self.embedding_size = bottom_to_output_size[self.bottom_neural_model]
            self.top_model = self.summerizer
        print('k')

        if word_guesser_type == WordGuessers.RNN_DECODER:
            self.words_guesser = RnnDecoderWordsGuesser(
                embedding_size=self.embedding_size,
                functions_subtokens_num=self.functions_vocabulary._vocabulary_size,
                eos_token=self.functions_vocabulary.eos_id(),
                bos_token=self.functions_vocabulary.bos_id())
        elif word_guesser_type == WordGuessers.LINEAR_DECODER:
            self.words_guesser = LinearDecoderWordsGuesser(
                embedding_size=self.embedding_size,
                functions_subtokens_num=self.functions_vocabulary._vocabulary_size)
        elif word_guesser_type == WordGuessers.DOUBLE_LINEAR_DECODER:
            self.words_guesser = LinearDecoderWordsGuesser(
                embedding_size=self.embedding_size,
                functions_subtokens_num=self.functions_vocabulary._vocabulary_size,
                linear_embedding_size=self.linear_embedding_size,
                double_linear_decoder=True)
        elif word_guesser_type == WordGuessers.TRIPLE_LINEAR_DECODER:
            self.words_guesser = LinearDecoderWordsGuesser(
                embedding_size=self.embedding_size,
                functions_subtokens_num=self.functions_vocabulary._vocabulary_size,
                linear_embedding_size=self.linear_embedding_size,
                double_linear_decoder=True,
                triple_linear_decoder=True)

        elif word_guesser_type == WordGuessers.GNEREIC_RNN_DECODER:
            self.words_guesser = nn.RNN()
        self.dire_evaluator = Evaluator()
        print('m')

        self.save_hyperparameters()
        print('p')

    def forward(self, x):
        # self.node_embedder.forward(function) returns node vector
        dire_data, graph_data = x
        plot_graph_data(graph_data)
        # dire_data_dict = self.Data_to_dict(dire_data)
        if self.bottom_neural_model == BottomNeuralModel.DIRE:
            dire_prediction_data = dire_data['prediction_target'][0]
            if not self.train_bottom_model:
                success = False
                try:
                    if dire_cache is not None and dire_cache.is_inside(x):
                        example_cache = dire_cache.get(x)
                        context_encoding_variable_encoding, predicted_dire_results = example_cache
                        print('loaded from cache')
                        success = True
                except:
                    print("didnt succeed loading from cache")
                    success = False
                if not success:
                    self.bottom_model.eval()
                    with torch.no_grad():
                        _, dire_context_encoding = self.bottom_model.forward(dire_data, dire_prediction_data)
                        context_encoding_variable_encoding = dire_context_encoding['variable_encoding']
                        predicted_dire_results = self.bottom_model.predict_tensors(dire_data)
                    if dire_cache is not None:
                        print('insert to cache')
                        if dire_cache is not None:# and not dire_cache.is_inside(x):
                            dire_cache.insert_to_cache(x, (
                                context_encoding_variable_encoding.clone().detach().cpu().numpy(),
                                predicted_dire_results))
            else:
                _, dire_context_encoding = self.bottom_model.forward(dire_data, dire_prediction_data)
                context_encoding_variable_encoding = dire_context_encoding['variable_encoding']
                predicted_dire_results = self.bottom_model.predict_tensors(dire_data)
        elif self.bottom_neural_model == BottomNeuralModel.CONSTANT:
            assert False, "not implemented, use 'BlankFunctionNamer'"
        elif self.bottom_neural_model == BottomNeuralModel.NERO:
            nero_ids = dire_data['nero_id'][0]
            context_encoding_variable_encoding, graph_data = get_relevant_nero_vectors_and_graph(nero_ids, graph_data)
            predicted_dire_results = None
        elif self.bottom_neural_model == BottomNeuralModel.NAMES:
            context_encoding_variable_encoding, graph_data = self.bottom_model.forward(graph_data)
            predicted_dire_results = None

        # dire encoding of variables - check accuracies
        # todo: change to function encoding instead of variable
        embedding = self.top_model.forward(context_encoding_variable_encoding, graph_data)
        guessed_words = self.words_guesser(embedding)
        flatten_logits = guessed_words.mean(dim=1).squeeze()

        return guessed_words, flatten_logits, predicted_dire_results, graph_data

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def calc_loss_and_top_1_acc(self, data_ast, data_graph, part='train', print_names_and_predictions=False):
        logits, flatten_logits, predicted_dire_results, data_graph = self.forward((data_ast, data_graph))
        y = data_graph[PROGRAM_GRAPH_FUNCTION_NODE_DEMANGLED_NAME_TOKENIZED].squeeze()
        should_validate_mask = data_graph[PROGRAM_GRAPH_FUNCTION_NODE_SHOULD_VALIDATE]
        should_validate_mask = (should_validate_mask == 1).view(-1, 1)
        # logits shape is: (functions, words from lstm, ids from vocab)
        # just as a reference to check, not important at all
        should_validate_logits_tag = logits * should_validate_mask.unsqueeze(-1)
        should_validate_logits_loss_tag = should_validate_logits_tag.mean(axis=1)[should_validate_mask.squeeze()]
        should_validate_y_tag = y * should_validate_mask
        should_validate_y_tag = should_validate_y_tag[should_validate_mask.squeeze()]
        # end of the reference

        # real calculations
        if should_validate_mask.shape[0] != 1:
            should_validate_mask = should_validate_mask.squeeze()
        else:
            should_validate_mask = torch.reshape(should_validate_mask, (1,))
        logits_summerized = logits.mean(axis=1)
        if len(y.shape) == 1:
            y = torch.reshape(y, (1, -1))

        should_validate_logits_loss = logits_summerized[should_validate_mask]
        should_validate_y = y[should_validate_mask]
        should_validate_logits = logits[should_validate_mask]

        assert (should_validate_y_tag == should_validate_y).all()
        assert (should_validate_logits_loss == should_validate_logits_loss_tag).all()

        if len(should_validate_logits_loss) > 0:
            loss = self.criterion(should_validate_logits_loss, should_validate_y)
        else:
            loss = torch.tensor([0.0], requires_grad=True).to(device=env_vars.torch_device)

        functions_evaluation_dict = self.dire_evaluator.functions_names_evaluation(should_validate_y,
                                                                                   should_validate_logits,
                                                                                   self.functions_id2word,
                                                                                   should_validate_mask,
                                                                                   words_sets_in_train=self.words_sets_in_train,
                                                                                   print_names_and_predictions=print_names_and_predictions)
        # eval per-word
        if predicted_dire_results is not None:
            _, dire_corpus_acc = self.dire_evaluator.evaluate_examples_batch(data_ast, predicted_dire_results)
        else:
            dire_corpus_acc = None

        if part == 'validation':
            relevant_logits = flatten_logits.argmax(dim=1)
            logger.info(relevant_logits)
            logger.info(
                [(i, self.functions_id2word[i]) for i in np.unique(relevant_logits.contiguous().cpu().numpy())])
        return loss, functions_evaluation_dict, dire_corpus_acc

    def generic_step(self, val_batch, name: str):
        data_graph, data_ast = val_batch
        if name == 'validation' or name == 'test':
            print_names_and_predictions = True
        else:
            print_names_and_predictions = False
        loss, functions_evaluation_dict, dire_corpus_acc = self.calc_loss_and_top_1_acc(data_ast, data_graph,
                                                                                        part=name,
                                                                                        print_names_and_predictions=print_names_and_predictions)
        self.log(name + '_loss', loss)
        for val_name, value in functions_evaluation_dict.items():
            if isinstance(value, tuple):
                dict_value = {'val': value[0], 'num': value[1]}
            else:
                dict_value = {'val': value, 'num': 1}
            self.log(name + '_' + val_name, dict_value, on_step=False, on_epoch=True, reduce_fx=weighted_tuples_mean)
        return loss

    def training_step(self, train_batch, batch_idx):
        loss = self.generic_step(train_batch, 'train')
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss = self.generic_step(val_batch, 'validation')
        return loss

    def test_step(self, test_batch, batch_idx):
        loss = self.generic_step(test_batch, 'test')
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
