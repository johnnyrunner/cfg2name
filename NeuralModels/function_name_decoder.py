# -*- coding: utf-8 -*-
"""
NLP From Scratch: Translation with a Sequence to Sequence Network and Attention
*******************************************************************************
**Author**: `Sean Robertson <https://github.com/spro/practical-pytorch>`_
"""

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from config import ATTN_DECODER_RNN_MAX_LENGTH, env_vars


class DecoderLSTM(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, dropout):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size=hidden_size, input_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.linear_layer = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax()

    def forward(self, input, hidden):
        # Switch batch and seq/layers dimension, so batch is second
        input = input.view(input.shape[1], input.shape[0], -1)
        hidden = hidden.view(hidden.shape[1], hidden.shape[0], -1)

        output = self.embedding(input).view(input.shape[0], input.shape[1], -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        # Switch batch and seq/layers dimension, so batch is first
        output = output.view(output.shape[1], output.shape[0], -1)
        hidden = hidden.view(hidden.shape[1], hidden.shape[0], -1)

        # Linear transformation
        output = self.linear_layer(output)
        output = self.softmax(output)

        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=env_vars.torch_device)

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers, dropout):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size=hidden_size, input_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.linear_layer = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax()

    def forward(self, input, hidden):
        # Switch batch and seq/layers dimension, so batch is second
        input = input.view(input.shape[1], input.shape[0], -1)
        hidden = hidden.view(hidden.shape[1], hidden.shape[0], -1)

        output = self.embedding(input).view(input.shape[0], input.shape[1], -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        # Switch batch and seq/layers dimension, so batch is first
        output = output.view(output.shape[1], output.shape[0], -1)
        hidden = hidden.view(hidden.shape[1], hidden.shape[0], -1)

        # Linear transformation
        output = self.linear_layer(output)
        output = self.softmax(output)

        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=env_vars.torch_device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=ATTN_DECODER_RNN_MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=env_vars.torch_device)