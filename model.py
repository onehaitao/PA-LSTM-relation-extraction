#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.6

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class PA_LSTM(nn.Module):
    def __init__(self, word_vec, class_num, config):
        super().__init__()
        self.word_vec = word_vec
        self.class_num = class_num
        self.device = config.device

        # hyper parameters and others
        self.max_len = config.max_len
        self.word_dim = config.word_dim
        self.pos_dim = config.pos_dim
        self.pos_dis = config.pos_dis
        self.dim = self.word_dim + 2 * self.pos_dim

        self.dropout_value = config.dropout
        self.hidden_size = config.hidden_size
        self.layers_num = config.layers_num

        # net structures and operations
        self.word_embedding = nn.Embedding.from_pretrained(
            embeddings=self.word_vec,
            freeze=False,
        )
        self.pos1_embedding = nn.Embedding(
            num_embeddings=2 * self.pos_dis + 3,
            embedding_dim=self.pos_dim
        )
        # self.pos2_embedding = nn.Embedding(
        #     num_embeddings=2 * self.pos_dis + 3,
        #     embedding_dim=self.pos_dim
        # )
        # self.pos1_embedding = nn.Embedding(
        #     num_embeddings=2 * self.max_len + 1,
        #     embedding_dim=self.pos_dim
        # )
        self.pos2_embedding = self.pos1_embedding
        # self.embedding_dropout = nn.Dropout(p=self.embedding_dropout)
        self.lstm = nn.LSTM(
            input_size=self.word_dim,
            hidden_size=self.hidden_size,
            num_layers=self.layers_num,
            bias=True,
            batch_first=True,
            dropout=self.dropout_value,
            bidirectional=False,

        )
        self.tanh = nn.Tanh()
        self.attention_weight = nn.Parameter(torch.randn(self.hidden_size))
        self.dropout = nn.Dropout(self.dropout_value)
        self.dense = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.class_num,
            bias=True
        )
        self.dense.bias.data.fill_(0)
        init.xavier_uniform_(self.dense.weight, gain=1)  # initialize linear layer

        self.att_len = 200
        # self.wh = nn.Parameter(torch.randn(self.hidden_size, self.att_len))
        # self.wq = nn.Parameter(torch.randn(self.hidden_size, self.att_len))
        # self.ws = nn.Parameter(torch.randn(self.pos_dim, self.att_len))
        # self.wo = nn.Parameter(torch.randn(self.pos_dim, self.att_len))
        # self.v = nn.Parameter(torch.randn(self.att_len))

        self.wh = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.att_len,
            bias=True
        )
        self.wq = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.att_len,
            bias=False
        )
        self.wso = nn.Linear(
            in_features=self.pos_dim*2,
            out_features=self.att_len,
            bias=False
        )
        self.v = nn.Linear(
            in_features=self.att_len,
            out_features=1,
            bias=True
        )

        self.wh.weight.data.normal_(std=0.001)
        self.wq.weight.data.normal_(std=0.001)
        self.wso.weight.data.normal_(std=0.001)
        self.v.weight.data.zero_()

        self.dense.bias.data.fill_(0)
        init.xavier_uniform_(self.dense.weight, gain=1)  # initialize linear layer
        self.pos1_embedding.weight.data.uniform_(-1.0, 1.0)

    def forward(self, data):
        tokens = data[:, 0, :].view(-1, self.max_len)
        pos1 = data[:, 1, :].view(-1, self.max_len)
        pos2 = data[:, 2, :].view(-1, self.max_len)
        mask = data[:, 3, :].view(-1, self.max_len)

        word_embedding = self.word_embedding(tokens)
        pos1_embedding = self.pos1_embedding(pos1)
        pos2_embedding = self.pos2_embedding(pos2)
        lengths = torch.sum(mask.gt(0), dim=-1)

        word_embedding = self.dropout(word_embedding)
        x = pack_padded_sequence(word_embedding, lengths, batch_first=True, enforce_sorted=False)
        h, (hn, _) = self.lstm(x)
        h, _ = pad_packed_sequence(h, batch_first=True, padding_value=0.0, total_length=self.max_len)

        q = self.dropout(hn[-1, :, :])
        h = self.dropout(h)
        # q = hn.permute(1, 0, 2).view(h.shape[0], 1, self.hidden_size)
        pos_embedding = torch.cat([pos1_embedding, pos2_embedding], dim=-1)
        state1 = self.wh(h.reshape(-1, self.hidden_size)).view(-1, self.max_len, self.att_len)
        state2 = self.wq(q.reshape(-1, self.hidden_size)).reshape(-1, 1, self.att_len).expand(-1, self.max_len, -1)
        state3 = self.wso(pos_embedding.reshape(-1, self.pos_dim*2)).view(-1, self.max_len, self.att_len)
        state = [state1, state2, state3]
        scores = self.v(self.tanh(sum(state)).view(-1, self.att_len)).view(-1, self.max_len)

        scores = scores.masked_fill_(mask.eq(0), float('-inf'))
        alpha = F.softmax(scores, dim=-1)
        z = alpha.unsqueeze(1).bmm(h).squeeze(1)
        # x = self.dropout(x)
        logits = self.dense(z)
        return logits
