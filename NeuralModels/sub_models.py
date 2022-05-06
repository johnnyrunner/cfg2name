from enum import Enum

import torch
from torch import nn as nn, Tensor
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, GATConv

from NeuralModels.function_name_decoder import DecoderRNN
from config import FUNCTION_NAME_DECODER_NUM_LAYERS, FUNCTION_NAME_DECODER_DROPOUT, env_vars, \
    ATTN_DECODER_RNN_MAX_LENGTH, DIRE_OUTPUT_SIZE, NERO_OUTPUT_SIZE, SIZE_OF_VOCAB


class BottomNeuralModel(Enum):
    DIRE = 'dire'
    NERO = 'nero'
    CONSTANT = 'no_matter_what_function_returns_conxtant_ones_vector'
    NAMES = 'all_names_but_some'


bottom_to_output_size = {
    BottomNeuralModel.DIRE: DIRE_OUTPUT_SIZE,
    BottomNeuralModel.CONSTANT: DIRE_OUTPUT_SIZE,
    BottomNeuralModel.NERO: NERO_OUTPUT_SIZE,
    BottomNeuralModel.NAMES: SIZE_OF_VOCAB,
}


class NeroOrDire(Enum):
    NERO = 'nero'
    Dire = 'dire'


class Id(nn.Module):
    def __init__(self, bottom_model):
        super(Id, self).__init__()
        self.bottom_model = bottom_model

    def inner_function_to_vector(self, function_variables_embedding):
        if self.bottom_model == BottomNeuralModel.DIRE or self.bottom_model == BottomNeuralModel.CONSTANT:
            output = function_variables_embedding.mean(-2).squeeze()
            assert output.shape[-1] == DIRE_OUTPUT_SIZE

        elif self.bottom_model == BottomNeuralModel.NERO:
            output = function_variables_embedding
            assert output.shape[-1] == NERO_OUTPUT_SIZE

        elif self.bottom_model == BottomNeuralModel.NAMES:
            output = function_variables_embedding.squeeze().float()
            assert output.shape[-1] == SIZE_OF_VOCAB
        # TODO: change to actual summary
        if len(output.shape) == 1:
            output = torch.reshape(output, (1, -1))
            print(output)
        return output

    def forward(self, function_variables_embedding, graph_data):
        inner_function = self.inner_function_to_vector(function_variables_embedding)
        assert len(inner_function.shape) == 2
        return inner_function


class RnnSummerizer(Id):
    def __init__(self, bottom_model, input_dim=DIRE_OUTPUT_SIZE, hidden_size=20, num_layers=1, bidirectional=True):
        super(RnnSummerizer, self).__init__(bottom_model)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.GRU(input_dim, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.bidirectional_factor = 2 if bidirectional else 1
        self.final_attn = nn.Linear(self.bidirectional_factor * num_layers * hidden_size, DIRE_OUTPUT_SIZE)

    def inner_function_to_vector(self, inner_function):
        # TODO: change to actual summary
        # print(inner_function.shape)
        # h_0 = torch.randn(2, 3, 20)
        hidden, _ = self.rnn(
            inner_function)  # hidden is (self.bidirectional_factor * self.num_layers, batch_size, hidden_size)
        hidden = hidden.permute(1, 0, 2)
        hidden = hidden[-1, :, :]
        hidden = hidden.reshape(-1, self.bidirectional_factor * self.num_layers * self.hidden_size)

        output = self.final_attn(hidden)
        # output = torch.sigmoid(output)
        return output

    def forward(self, function_variables_embedding, graph_data):
        inner_function = self.inner_function_to_vector(function_variables_embedding)
        assert inner_function.shape[-1] == DIRE_OUTPUT_SIZE
        assert len(inner_function.shape) == 2
        return inner_function


class Summerizer(Enum):
    ID = 'vector_itself_or_average_of_the_vectors'
    RNN_SUMMERIZER = 'rnn_with_attention_to_summerize_the_whole_sentence'


class GCN(nn.Module):
    def __init__(self, bottom_model, embedding_size, summerizer=None, two_layered=False, **kwargs):
        if summerizer is None:
            summerizer = Id(bottom_model)
        super(GCN, self).__init__()
        self.summerizer = summerizer
        self.conv1 = GCNConv(bottom_to_output_size[bottom_model], embedding_size, node_dim=-2)
        self.two_layered = two_layered
        if self.two_layered:
            print('using two layered GCN')
            self.conv2 = GCNConv(embedding_size, embedding_size, node_dim=-2)

    def forward(self, dire_function_variables_embedding, graph_data):
        x, edge_index = dire_function_variables_embedding, graph_data.edge_index
        inner_function = self.summerizer.forward(x, edge_index)
        x = self.conv1(inner_function, edge_index)
        x = F.relu(x)
        if self.two_layered:
            x = self.conv2(x, edge_index)
            x = F.relu(x)
        return x


class GAT_model(nn.Module):
    def __init__(self, bottom_model, embedding_size, summerizer=None, two_layered=False, **kwargs):
        if summerizer is None:
            summerizer = Id(bottom_model)
        super(GAT_model, self).__init__()
        self.summerizer = summerizer
        self.two_layered = two_layered
        self.gat1 = GATConv(bottom_to_output_size[bottom_model], embedding_size)
        if self.two_layered:
            print('using two layered GAT')
            self.gat2 = GATConv(embedding_size, embedding_size)

    def forward(self, dire_function_variables_embedding, graph_data):
        x, edge_index = dire_function_variables_embedding, graph_data.edge_index
        inner_function = self.summerizer.forward(x, edge_index)
        x = self.gat1(inner_function, edge_index.type(torch.long))
        x = F.relu(x)
        if self.two_layered:
            x = self.gat2(x, edge_index)
        return x


class TopNeuralModels(Enum):
    ID_DIRE_CHECK = 'dire_without_anything_on_top'
    ID_NERO_CHECK = 'nero_without_anything_on_top'
    GCN = 'gcn_on_top'
    GAT = 'gat_on_top'


class RnnDecoderWordsGuesser(nn.Module):
    def __init__(self, embedding_size, functions_subtokens_num, eos_token, bos_token):
        super(RnnDecoderWordsGuesser, self).__init__()
        self.decoder = DecoderRNN(
            hidden_size=embedding_size,
            output_size=functions_subtokens_num,
            num_layers=FUNCTION_NAME_DECODER_NUM_LAYERS,
            dropout=FUNCTION_NAME_DECODER_DROPOUT)

        self.eos_token = eos_token
        self.bos_token = bos_token
        self.functions_subtokens_num = functions_subtokens_num
        self.bos_input = torch.zeros((functions_subtokens_num))
        if env_vars.use_gpu:
            self.bos_input = self.bos_input.cuda()
        self.bos_input[bos_token] = 1

    def forward(self, hidden_state_init: Tensor):
        # Our hidden state is our real input (small size), and our 'input' is sos.
        batch_size = hidden_state_init.shape[0]

        # Input dimensions: (batch_size, seq_length = 1, input_size)
        decoder_input = torch.tensor([[self.bos_token]], device=env_vars.torch_device).repeat(batch_size, 1).view(
            batch_size, 1, -1)
        # decoder_input = self.bos_input.repeat(batch_size, 1).view(batch_size, FUNCTION_NAME_DECODER_NUM_LAYERS, -1)

        # Hidden / cell state dimensions: (batch_size, num_layers, hidden_size)
        decoder_hidden = hidden_state_init.view(batch_size, FUNCTION_NAME_DECODER_NUM_LAYERS, -1)
        # cell_state_init = cell_state_init.view(FUNCTION_NAME_DECODER_NUM_LAYERS, batch_size, -1)

        total_output = torch.zeros((batch_size, 0, self.functions_subtokens_num), device=env_vars.torch_device)

        for i in range(ATTN_DECODER_RNN_MAX_LENGTH):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            total_output = torch.cat((total_output, decoder_output.view(batch_size, 1, -1)), dim=1)

            if (total_output == self.eos_token).any(dim=1).all():
                break

        return total_output


class LinearDecoderWordsGuesser(nn.Module):
    def __init__(self, embedding_size, functions_subtokens_num, double_linear_decoder=False, linear_embedding_size=None,
                 triple_linear_decoder=False):
        super(LinearDecoderWordsGuesser, self).__init__()
        self.double_linear_decoder = double_linear_decoder
        self.triple_linear_decoder = triple_linear_decoder
        self.functions_subtokens_num = functions_subtokens_num
        if self.double_linear_decoder:
            self.double_linear_layer = nn.Linear(embedding_size, linear_embedding_size)
            self.linear_layer = nn.Linear(linear_embedding_size,
                                          (self.functions_subtokens_num * ATTN_DECODER_RNN_MAX_LENGTH))
        else:
            self.linear_layer = nn.Linear(embedding_size, (self.functions_subtokens_num * ATTN_DECODER_RNN_MAX_LENGTH))

        if self.triple_linear_decoder:
            self.third_linear_layer = nn.Linear(linear_embedding_size, linear_embedding_size)

    def forward(self, embedding):
        if self.double_linear_decoder:
            embedding = self.double_linear_layer(embedding)
            embedding = F.relu(embedding)
        if self.triple_linear_decoder:
            embedding = self.third_linear_layer(embedding)
            embedding = F.relu(embedding)

        word_guess = self.linear_layer(embedding)
        word_guess = word_guess.reshape((-1, ATTN_DECODER_RNN_MAX_LENGTH, self.functions_subtokens_num))
        # word_guess = self.softmax(word_guess)
        return word_guess
