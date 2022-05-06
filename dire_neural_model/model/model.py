import pprint
import sys
from typing import List, Dict, Tuple, Iterable

import torch
import torch.nn as nn
from torch import Tensor

from config import env_vars
from dire_neural_model.utils import nn_util, util
from dire_neural_model.utils.ast import AbstractSyntaxTree
from dire_neural_model.model.decoder import Decoder
from dire_neural_model.model.recurrent_subtoken_decoder import RecurrentSubtokenDecoder
from dire_neural_model.model.attentional_recurrent_subtoken_decoder import AttentionalRecurrentSubtokenDecoder
from dire_neural_model.model.recurrent_decoder import RecurrentDecoder
from dire_neural_model.model.simple_decoder import SimpleDecoder
from dire_neural_model.model.encoder import Encoder
from dire_neural_model.model.hybrid_encoder import HybridEncoder
from dire_neural_model.model.sequential_encoder import SequentialEncoder
from dire_neural_model.model.graph_encoder import GraphASTEncoder
from dire_neural_model.utils.graph import PackedGraph
from dire_neural_model.utils.dataset import Batcher, Example
from dire_neural_model.utils.dire_vocab import SAME_VARIABLE_TOKEN
import numpy as np


class ConstantModel(nn.Module):
    def forward(self, source_asts: Dict[str, torch.Tensor], prediction_target: Dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.ones((source_asts.shape[0], 1))

    def predict(self, examples: List[Example]):
        assert False, "should not get here, this is not a real model."

    def predict_tensors(self, examples: Tensor):
        assert False, "should not get here, this is not a real model."


class NamesModel(nn.Module):
    def __init__(self, number_not_real=1, give_answers=False, portion_not_real=None):
        super(NamesModel, self).__init__()
        self.number_not_real = number_not_real
        self.give_answers = give_answers
        self.portion_not_real = portion_not_real

    def forward(self, graph_data):
        tokenized_names = graph_data['demangled_function_name_tokenized']
        original_should_validate = torch.nonzero(graph_data['should_validate']).cpu().detach().numpy().squeeze()
        if self.portion_not_real:
            number_not_real = int(len(original_should_validate) * self.portion_not_real) + 1
        else:
            number_not_real = self.number_not_real
        should_validate_indices = np.random.choice(original_should_validate, number_not_real)
        names_model_mask = torch.zeros(size=(tokenized_names.shape[0],))
        names_model_mask[should_validate_indices] = 1
        names_model_mask = (names_model_mask[torch.randperm(len(names_model_mask))] > 0)
        inputs_prediction_target = tokenized_names.clone()
        inputs_prediction_target[names_model_mask] = 0
        outputs_prediction_target = tokenized_names.clone()
        outputs_prediction_target[~names_model_mask] = 0
        # assert (not (outputs_prediction_target.sum(1) > 0).all()) and (
        #     not (inputs_prediction_target.sum(1) > 0).all()) and (
        #                (inputs_prediction_target.sum(1) + outputs_prediction_target.sum(1)) > 0).all(), (inputs_prediction_target.sum(1), outputs_prediction_target.sum(1), number_not_real)
        # assert graph_data['should_validate'].shape == names_model_mask.shape
        graph_data['should_validate'] = names_model_mask.to(device=env_vars.torch_device)
        if self.give_answers:
            return outputs_prediction_target.to(device=env_vars.torch_device), graph_data
        return inputs_prediction_target.to(device=env_vars.torch_device), graph_data

    def predict(self, examples: List[Example]):
        assert False, "should not get here, this is not a real model."

    def predict_tensors(self, examples: Tensor):
        assert False, "should not get here, this is not a real model."


class RenamingModel(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, should_decode: bool = False):
        super(RenamingModel, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.config: Dict = None
        self.should_decode = should_decode

    @property
    def vocab(self):
        return self.encoder.vocab

    @property
    def batcher(self):
        if not hasattr(self, '_batcher'):
            _batcher = Batcher(self.config)
            setattr(self, '_batcher', _batcher)

        return self._batcher

    @property
    def device(self):
        return self.encoder.device

    @classmethod
    def default_params(cls):
        return {
            'train': {
                'unchanged_variable_weight': 1.0,
                'max_epoch': 30,
                'patience': 5
            },
            'decoder': {
                'type': 'SimpleDecoder'
            }
        }

    @classmethod
    def build(cls, config, should_decode=False):
        params = util.update(cls.default_params(), config)
        encoder = globals()[config['encoder']['type']].build(config['encoder'])
        decoder = globals()[config['decoder']['type']].build(config['decoder'])

        model = cls(encoder, decoder, should_decode=should_decode)
        params = util.update(params, {'encoder': encoder.config,
                                      'decoder': decoder.config})
        model.config = params
        model.decoder.encoder = encoder  # give the decoder a reference to the encoder

        # assign batcher to sub-modules
        encoder.batcher = model.batcher
        decoder.batcher = model.batcher

        print('Current Configuration:', file=sys.stderr)
        pp = pprint.PrettyPrinter(indent=2, stream=sys.stderr)
        pp.pprint(model.config)

        return model

    def forward(self, source_asts: Dict[str, torch.Tensor], prediction_target: Dict[str, torch.Tensor]) -> Tuple[
        torch.Tensor, Dict]:
        """
        Given a batch of decompiled abstract syntax trees, and the gold-standard renaming of variable nodes,
        compute the log-likelihood of the gold-standard renaming for training

        Arg:
            source_asts: a list of ASTs
            variable_name_maps: mapping of decompiled variable names to its renamed values

        Return:
            a tensor of size (batch_size) denoting the log-likelihood of renamings
        """

        # src_ast_encoding: (batch_size, max_ast_node_num, node_encoding_size)
        # src_ast_mask: (batch_size, max_ast_node_num)
        context_encoding = self.encoder(source_asts)

        # (batch_size, variable_num, vocab_size) or (prediction_node_num, vocab_size)
        if self.should_decode:
            var_name_log_probs = self.decoder(context_encoding, prediction_target)
            #
            result = self.decoder.get_target_log_prob(var_name_log_probs, prediction_target, context_encoding)

            tgt_var_name_log_prob = result['tgt_var_name_log_prob']
            tgt_weight = prediction_target['variable_tgt_name_weight']
            weighted_log_prob = tgt_var_name_log_prob * tgt_weight

            ast_log_probs = weighted_log_prob.sum(dim=-1) / prediction_target[
                'target_variable_encoding_indices_mask'].sum(-1)
            result['batch_log_prob'] = ast_log_probs

            return result, context_encoding

        return context_encoding

    def decode_dataset(self, dataset, batch_size=4096) -> Iterable[Tuple[Example, Dict]]:
        with torch.no_grad():
            data_iter = dataset.batch_iterator(batch_size=batch_size, train=False, progress=False,
                                               config=self.config)
            was_training = self.training
            self.eval()

            for batch in data_iter:
                rename_results = self.decoder.predict([e._ast for e in batch.examples], self.encoder)
                for example, rename_result in zip(batch.examples, rename_results):
                    yield example, rename_result

    def predict(self, examples: List[Example]):
        return self.decoder.predict(examples, self.encoder)

    def predict_tensors(self, examples: Tensor):
        # kavitzky
        return self.decoder.predict_tensors(examples, self.encoder)

    def save(self, model_path, **kwargs):
        params = {
            'config': self.config,
            'state_dict': self.state_dict(),
            'kwargs': kwargs
        }

        torch.save(params, model_path)

    @classmethod
    def load(cls, model_path, use_cuda=False, new_config=None, should_decode=True) -> 'RenamingModel':
        device = torch.device("cuda:0" if use_cuda else "cpu")
        params = torch.load(model_path, map_location=lambda storage, loc: storage)

        config = params['config']
        if new_config:
            config = util.update(config, new_config)

        kwargs = params['kwargs'] if params['kwargs'] is not None else dict()

        model = cls.build(config, should_decode=should_decode, **kwargs)
        model.load_state_dict(params['state_dict'], strict=False)
        model = model.to(device)
        model.eval()

        return model
