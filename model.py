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

        # self.embedding_dropout = config.embedding_dropout
        # self.lstm_dropout = config.lstm_dropout
        # self.liner_dropout = config.liner_dropout
        self.dropout_value = config.dropout
        self.hidden_size = config.hidden_size
        self.layers_num = config.layers_num

        # net structures and operations
        self.word_embedding = nn.Embedding.from_pretrained(
            embeddings=self.word_vec,
            freeze=False,
        )
        # self.pos1_embedding = nn.Embedding(
        #     num_embeddings=2 * self.pos_dis + 3,
        #     embedding_dim=self.pos_dim
        # )
        # self.pos2_embedding = nn.Embedding(
        #     num_embeddings=2 * self.pos_dis + 3,
        #     embedding_dim=self.pos_dim
        # )
        self.pos1_embedding = nn.Embedding(
            num_embeddings=2 * self.max_len + 1,
            embedding_dim=self.pos_dim
        )
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
        # x = self.input(tokens, pos1, pos2)
        # x = self.lstm_layer(x, mask)
        # x = self.attention_layer(x, mask)
        word_embedding = self.word_embedding(tokens)
        pos1_embedding = self.pos1_embedding(pos1)
        pos2_embedding = self.pos2_embedding(pos2)
        lengths = torch.sum(mask.gt(0), dim=-1)

        # h0 = c0 = torch.zeros([self.layers_num, tokens.shape[0], self.hidden_size],
        #                       device=self.device,
        #                       requires_grad=False)
        word_embedding = self.dropout(word_embedding)
        x = pack_padded_sequence(
            word_embedding, lengths, batch_first=True, enforce_sorted=False)
        # h, (hn, _) = self.lstm(x, (h0, c0))
        h, (hn, _) = self.lstm(x)
        # print(h.shape)
        # print(hn.shape)
        # hn 2*B*L
        h, _ = pad_packed_sequence(
            h, batch_first=True, padding_value=0.0, total_length=self.max_len)

        q = self.dropout(hn[-1, :, :])
        h = self.dropout(h)
        # q = hn.permute(1, 0, 2).view(h.shape[0], 1, self.hidden_size)
        # state1 = torch.matmul(h, self.wh)  # B*L*A
        # state2 = torch.matmul(q, self.wq)  # B*1*A
        # state3 = torch.matmul(pos1_embedding, self.ws)  # B*L*A
        # state4 = torch.matmul(pos2_embedding, self.wo)  # B*L*A
        # u = self.tanh(state1 + state2 + state3 + state4).matmul(self.v.view(-1, 1))  # B*L*1
        # u = u.view(u.shape[0], u.shape[1])
        # alpha = F.softmax(u, dim=-1).view(u.shape[0], u.shape[1], 1)  # B*L
        # z = torch.mul(h, alpha.expand(-1, -1, h.shape[-1]))  # B*L*H
        # z = torch.sum(z, dim=1)  # B*H
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


class CNN(nn.Module):
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
        self.filter_num = config.filter_num
        self.window = config.window
        self.keep_prob = config.dropout

        self.dim = self.word_dim + 2 * self.pos_dim

        # net structures and operations
        self.word_embedding = nn.Embedding.from_pretrained(
            embeddings=self.word_vec,
            freeze=False,
        )
        self.pos1_embedding = nn.Embedding(
            num_embeddings=2 * self.pos_dis + 3,
            embedding_dim=self.pos_dim
        )
        self.pos2_embedding = nn.Embedding(
            num_embeddings=2 * self.pos_dis + 3,
            embedding_dim=self.pos_dim
        )
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=self.filter_num,
            kernel_size=(self.window, self.dim),
            stride=(1, 1),
            bias=True,
            padding=(1, 0),  # same padding
            padding_mode='zeros'
        )
        self.maxpool = nn.MaxPool2d((self.max_len, 1))
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(self.keep_prob)
        self.dense = nn.Linear(
            in_features=self.filter_num,
            out_features=self.class_num,
            bias=True
        )

    def input(self, tokens, pos1, pos2):
        word_embedding = self.word_embedding(tokens)
        pos1_embedding = self.pos1_embedding(pos1)
        pos2_embedding = self.pos2_embedding(pos2)
        x = torch.cat(tensors=[word_embedding, pos1_embedding, pos2_embedding], dim=-1)
        return x

    def convolution(self, x, mask):
        x = x.unsqueeze(dim=1)
        x = self.conv(x)
        x = x.view(-1, self.filter_num, self.max_len)

        # mask, remove the effect of 'PAD'
        mask = mask.unsqueeze(dim=1)
        mask = mask.expand(mask.shape[0], self.filter_num, mask.shape[-1])
        x = x.masked_fill_(mask.eq(0), float('-inf'))
        x = x.unsqueeze(dim=-1)
        return x

    def forward(self, data):
        tokens = data[:, 0, :].view(-1, self.max_len)
        pos1 = data[:, 1, :].view(-1, self.max_len)
        pos2 = data[:, 2, :].view(-1, self.max_len)
        mask = data[:, 3, :].view(-1, self.max_len)
        x = self.input(tokens, pos1, pos2)
        x = self.convolution(x, mask)
        x = self.maxpool(x)
        x = x.view(-1, self.filter_num)
        x = self.tanh(x)
        x = self.dropout(x)
        out = self.dense(x)
        return out


class PCNN(nn.Module):
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
        self.filter_num = config.filter_num
        self.window = config.window
        self.keep_prob = config.dropout

        self.dim = self.word_dim + 2 * self.pos_dim

        # net structures and operations
        self.word_embedding = nn.Embedding.from_pretrained(
            embeddings=self.word_vec,
            freeze=False,
        )
        self.pos1_embedding = nn.Embedding(
            num_embeddings=2 * self.pos_dis + 3,
            embedding_dim=self.pos_dim
        )
        self.pos2_embedding = nn.Embedding(
            num_embeddings=2 * self.pos_dis + 3,
            embedding_dim=self.pos_dim
        )
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=self.filter_num,
            kernel_size=(self.window, self.dim),
            stride=(1, 1),
            bias=True,
            padding=(1, 0),  # same padding
            padding_mode='zeros'
        )
        self.maxpool = nn.MaxPool2d((self.max_len, 1))
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(self.keep_prob)
        self.dense = nn.Linear(
            in_features=self.filter_num*3,
            out_features=self.class_num,
            bias=True
        )

        # mask operation for pcnn
        self.mask_embedding = nn.Embedding(4, 3)
        masks = torch.tensor([[0, 0, 0], [100, 0, 0], [0, 100, 0], [0, 0, 100]])
        self.mask_embedding.weight.data.copy_(masks)
        self.mask_embedding.weight.requires_grad = False

        nn.init.xavier_normal_(self.dense.weight)
        nn.init.constant_(self.dense.bias, 0.)
        nn.init.xavier_normal_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0.)

    def input(self, tokens, pos1, pos2):
        word_embedding = self.word_embedding(tokens)
        pos1_embedding = self.pos1_embedding(pos1)
        pos2_embedding = self.pos2_embedding(pos2)
        x = torch.cat(tensors=[word_embedding, pos1_embedding, pos2_embedding], dim=-1)
        return x

    def convolution(self, x, mask):
        x = x.unsqueeze(dim=1)
        x = self.conv(x)
        x = x.view(-1, self.filter_num, self.max_len)

        # mask, remove the effect of 'PAD'
        mask = mask.unsqueeze(dim=1)
        mask = mask.expand(mask.shape[0], self.filter_num, mask.shape[-1])
        x = x.masked_fill_(mask.eq(0), float('-inf'))
        x = x.unsqueeze(dim=-1)
        return x

    def piece_maxpool(self, x, mask):
        x = x.permute(0, 2, 1, 3)
        mask_embed = self.mask_embedding(mask)
        mask_embed = mask_embed.unsqueeze(dim=-2)
        x = x + mask_embed
        x = torch.max(x, dim=1)[0] - 100
        x = x.view(x.shape[0], -1)
        return x

    def forward(self, data):
        tokens = data[:, 0, :].view(-1, self.max_len)
        pos1 = data[:, 1, :].view(-1, self.max_len)
        pos2 = data[:, 2, :].view(-1, self.max_len)
        mask = data[:, 3, :].view(-1, self.max_len)
        x = self.input(tokens, pos1, pos2)
        x = self.convolution(x, mask)
        x = self.piece_maxpool(x, mask)
        # x = self.maxpool(x)
        # x = x.view(-1, self.filter_num)
        x = self.tanh(x)
        x = self.dropout(x)
        out = self.dense(x)
        return out


class BiLSTM(nn.Module):
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

        # self.embedding_dropout = config.embedding_dropout
        # self.lstm_dropout = config.lstm_dropout
        # self.liner_dropout = config.liner_dropout
        self.dropout_value = config.dropout
        self.hidden_size = config.hidden_size

        # net structures and operations
        self.word_embedding = nn.Embedding.from_pretrained(
            embeddings=self.word_vec,
            freeze=False,
        )
        self.pos1_embedding = nn.Embedding(
            num_embeddings=2 * self.pos_dis + 3,
            embedding_dim=self.pos_dim
        )
        self.pos2_embedding = nn.Embedding(
            num_embeddings=2 * self.pos_dis + 3,
            embedding_dim=self.pos_dim
        )
        # self.embedding_dropout = nn.Dropout(p=self.embedding_dropout)
        self.bi_lstm = nn.LSTM(
            input_size=self.dim,
            hidden_size=self.hidden_size,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=True,

        )
        self.tanh = nn.Tanh()
        # self.lstm_dropout = nn.Dropout(p=self.lstm_dropout)
        # self.attention_weight = nn.Parameter(torch.randn(self.hidden_size))
        # self.liner_dropout = nn.Dropout(p=self.liner_dropout)
        self.dropout = nn.Dropout(self.dropout_value)
        self.dense = nn.Linear(
            in_features=self.hidden_size*2,
            out_features=self.class_num,
            bias=True
        )

    def input(self, tokens, pos1, pos2):
        word_embedding = self.word_embedding(tokens)
        pos1_embedding = self.pos1_embedding(pos1)
        pos2_embedding = self.pos2_embedding(pos2)
        x = torch.cat(
            tensors=[word_embedding, pos1_embedding, pos2_embedding], dim=-1)
        return x

    def lstm_layer(self, x, mask):
        lengths = torch.sum(mask.gt(0), dim=-1)
        x = pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False)
        _, (x, _) = self.bi_lstm(x)
        x = x.permute(1, 0, 2).reshape(-1, self.hidden_size*2)
        return x

    def forward(self, data):
        tokens = data[:, 0, :].view(-1, self.max_len)
        pos1 = data[:, 1, :].view(-1, self.max_len)
        pos2 = data[:, 2, :].view(-1, self.max_len)
        mask = data[:, 3, :].view(-1, self.max_len)
        x = self.input(tokens, pos1, pos2)
        # x = self.embedding_dropout(x)
        x = self.lstm_layer(x, mask)
        # x = self.liner_dropout(x)
        x = self.dropout(x)
        logits = self.dense(x)
        return logits


class ATT_BLSTM(nn.Module):
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

        # self.embedding_dropout = config.embedding_dropout
        # self.lstm_dropout = config.lstm_dropout
        # self.liner_dropout = config.liner_dropout
        self.dropout_value = config.dropout
        self.hidden_size = config.hidden_size

        # net structures and operations
        self.word_embedding = nn.Embedding.from_pretrained(
            embeddings=self.word_vec,
            freeze=False,
        )
        self.pos1_embedding = nn.Embedding(
            num_embeddings=2 * self.pos_dis + 3,
            embedding_dim=self.pos_dim
        )
        self.pos2_embedding = nn.Embedding(
            num_embeddings=2 * self.pos_dis + 3,
            embedding_dim=self.pos_dim
        )
        # self.embedding_dropout = nn.Dropout(p=self.embedding_dropout)
        self.bi_lstm = nn.LSTM(
            input_size=self.dim,
            hidden_size=self.hidden_size,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=True,

        )
        self.tanh = nn.Tanh()
        # self.lstm_dropout = nn.Dropout(p=self.lstm_dropout)
        self.attention_weight = nn.Parameter(torch.randn(self.hidden_size))
        # self.liner_dropout = nn.Dropout(p=self.liner_dropout)
        self.dropout = nn.Dropout(self.dropout_value)
        self.dense = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.class_num,
            bias=True
        )

    def input(self, tokens, pos1, pos2):
        word_embedding = self.word_embedding(tokens)
        pos1_embedding = self.pos1_embedding(pos1)
        pos2_embedding = self.pos2_embedding(pos2)
        x = torch.cat(
            tensors=[word_embedding, pos1_embedding, pos2_embedding], dim=-1)
        return x

    def lstm_layer(self, x, mask):
        lengths = torch.sum(mask.gt(0), dim=-1)
        x = pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False)
        x, (_, _) = self.bi_lstm(x)
        x, _ = pad_packed_sequence(
            x, batch_first=True, padding_value=0.0, total_length=self.max_len)
        x = x.view(-1, self.max_len, 2, self.hidden_size)
        x = torch.sum(x, dim=2)
        return x

    def attention_layer(self, x, mask):
        att = self.attention_weight.view(
            1, -1, 1).expand(x.shape[0], -1, -1)  # B*C*1
        att_score = torch.bmm(self.tanh(x), att)  # B*L*C  *  B*C*1 -> B*L*1

        # mask, remove the effect of 'PAD'
        mask = mask.unsqueeze(dim=-1)
        att_score = att_score.masked_fill_(mask.eq(0), float('-inf'))
        att_weight = F.softmax(att_score, dim=1)  # B*L*1

        reps = torch.bmm(x.transpose(1, 2), att_weight).squeeze(
            dim=-1)  # B*C*L *  B*L*1 -> B*C*1 -> B*C
        x = self.tanh(reps)  # B*C
        return x

    def forward(self, data):
        tokens = data[:, 0, :].view(-1, self.max_len)
        pos1 = data[:, 1, :].view(-1, self.max_len)
        pos2 = data[:, 2, :].view(-1, self.max_len)
        mask = data[:, 3, :].view(-1, self.max_len)
        x = self.input(tokens, pos1, pos2)
        # x = self.embedding_dropout(x)
        x = self.lstm_layer(x, mask)
        # x = self.lstm_dropout(x)
        x = self.attention_layer(x, mask)
        # x = self.liner_dropout(x)
        x = self.dropout(x)
        logits = self.dense(x)
        return logits
