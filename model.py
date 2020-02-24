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

        self.dropout_value = config.dropout
        self.word_dropout_value = config.word_dropout
        self.hidden_size = config.hidden_size
        self.layers_num = config.layers_num
        self.att_len = config.att_len

        # net structures and operations
        self.word_embedding = nn.Embedding.from_pretrained(
            embeddings=self.word_vec,
            freeze=False,
        )
        self.pos1_embedding = nn.Embedding(
            num_embeddings=2 * self.max_len - 1,
            embedding_dim=self.pos_dim
        )
        self.pos2_embedding = nn.Embedding(
            num_embeddings=2 * self.max_len - 1,
            embedding_dim=self.pos_dim
        )

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
        self.dropout = nn.Dropout(self.dropout_value)
        self.word_dropout = nn.Dropout(self.word_dropout_value)
        self.dense = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.class_num,
            bias=True
        )

        self.wh = nn.Parameter(torch.randn(1, self.hidden_size, self.att_len))
        self.wq = nn.Parameter(torch.randn(1, self.hidden_size, self.att_len))
        self.ws = nn.Parameter(torch.randn(1, self.pos_dim, self.att_len))
        self.wo = nn.Parameter(torch.randn(1, self.pos_dim, self.att_len))
        self.v = nn.Parameter(torch.randn(1, self.att_len, 1))
        self.wb = nn.Parameter(torch.zeros(1, 1, self.att_len))
        self.vb = nn.Parameter(torch.zeros(1, 1, 1))

        # initialize weight
        self.pos1_embedding.weight.data.uniform_(-1.0, 1.0)
        self.pos2_embedding.weight.data.uniform_(-1.0, 1.0)

        self.wh.data.normal_(std=0.001)
        self.wq.data.normal_(std=0.001)
        self.ws.data.normal_(std=0.001)
        self.wo.data.normal_(std=0.001)

        init.xavier_uniform_(self.dense.weight, gain=1)
        self.dense.bias.data.fill_(0)

    def word_dropout_layer(self, token, mask):
        # <UNK>' word2id is 1
        unk_mask = torch.ones(size=token.shape, device=self.device, requires_grad=False)
        unk_mask = self.word_dropout(unk_mask)
        # the <PAD> will not be replaced by <UNK>
        unk_mask = unk_mask.masked_fill_(mask.eq(0), 1)
        token = token.masked_fill_(unk_mask.eq(0), 1)
        return token

    def encoder_layer(self, token, pos1, pos2):
        word_emb = self.word_embedding(token)  # B * L * word_dim
        pos1_emb = self.pos1_embedding(pos1)  # B * L * pos_dim
        pos2_emb = self.pos2_embedding(pos2)  # B * L * pos_dim
        return word_emb, pos1_emb, pos2_emb

    def lstm_layer(self, word_emb, mask):
        word_emb = self.dropout(word_emb)
        lengths = torch.sum(mask.gt(0), dim=-1)
        x = pack_padded_sequence(word_emb, lengths, batch_first=True, enforce_sorted=False)
        h, (_, _) = self.lstm(x)
        h, _ = pad_packed_sequence(h, batch_first=True, padding_value=0.0, total_length=self.max_len)
        return h  # B * L * H

    def attention_layer(self, h, pos1_emb, pos2_emb, mask):
        q = h[:, -1, :].view(-1, 1, self.hidden_size)  # B * 1 * H
        batch_size = q.shape[0]
        h = self.dropout(h)
        q = self.dropout(q)

        wh = self.wh.expand(batch_size, -1, -1)  # B * H * A
        wq = self.wq.expand(batch_size, -1, -1)  # B * H * A
        ws = self.ws.expand(batch_size, -1, -1)  # B * pos_dim * A
        wo = self.wo.expand(batch_size, -1, -1)  # B * pos_dim * A
        v = self.v.expand(batch_size, -1, 1)  # B * A * 1
        wb = self.wb.expand(batch_size, self.max_len, -1)  # B * L * A
        vb = self.vb.expand(batch_size, self.max_len, -1)

        s1 = torch.bmm(h, wh)  # B * L * A
        s2 = torch.bmm(q.expand(-1, self.max_len, -1), wq)
        s3 = torch.bmm(pos1_emb, ws)
        s4 = torch.bmm(pos2_emb, wo)
        s = self.tanh(s1 + s2 + s3 + s4 + wb)  # B * L * A

        score = (torch.bmm(s, v) + vb).view(-1, self.max_len)  # B * L * 1 -> B * L
        score = score.masked_fill_(mask.eq(0), float('-inf'))
        alpha = F.softmax(score, dim=-1).unsqueeze(1)  # B * 1 * L
        z = torch.bmm(alpha, h).view(-1, self.hidden_size)  # B * H
        return z

    def forward(self, data):
        token = data[:, 0, :].view(-1, self.max_len)
        pos1 = data[:, 1, :].view(-1, self.max_len)
        pos2 = data[:, 2, :].view(-1, self.max_len)
        mask = data[:, 3, :].view(-1, self.max_len)

        token = self.word_dropout_layer(token, mask)
        word_emb, pos1_emb, pos2_emb = self.encoder_layer(token, pos1, pos2)
        h = self.lstm_layer(word_emb, mask)
        z = self.attention_layer(h, pos1_emb, pos2_emb, mask)
        logits = self.dense(z)
        return logits
