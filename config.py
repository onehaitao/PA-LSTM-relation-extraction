#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.6

import argparse
import torch
import os
import random
import json
import numpy as np


class Config(object):
    def __init__(self):
        # get init config
        args = self.__get_config()
        for key in args.__dict__:
            setattr(self, key, args.__dict__[key])

        # select device
        self.device = None
        if self.cuda >= 0 and torch.cuda.is_available():
            self.device = torch.device('cuda:{}'.format(self.cuda))
        else:
            self.device = torch.device('cpu')

        # determine the model name and model dir
        if self.model_name is None:
            self.model_name = 'PA-LSTM'
        self.model_dir = os.path.join(self.output_dir, self.model_name)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        # backup data
        self.__config_backup(args)

        # set the random seed
        self.__set_seed(self.seed)

    def __get_config(self):
        parser = argparse.ArgumentParser()
        parser.description = 'config for models'

        # several key selective parameters
        parser.add_argument('--data_dir', type=str,
                            default='/data6/htwang/resource/data/TACRED',
                            help='dir to load data')
        parser.add_argument('--output_dir', type=str,
                            default='./output',
                            help='dir to save output')

        # word embedding
        parser.add_argument('--embedding_path', type=str,
                            default='/data6/htwang/resource/embedding/glove/glove.6B.200d.txt',
                            help='pre_trained word embedding')

        parser.add_argument('--word_dim', type=int,
                            default=200,
                            help='dimension of word embedding')

        # train settings
        parser.add_argument('--model_name', type=str,
                            default=None,
                            help='model name')
        parser.add_argument('--mode', type=int,
                            default='1',
                            choices=[0, 1],
                            help='running mode: 1 for training; otherwise testing')
        parser.add_argument('--seed', type=int,
                            default=1234,
                            help='random seed')
        parser.add_argument('--cuda', type=int,
                            default=0,
                            help='num of gpu device, if -1, select cpu')
        parser.add_argument('--epoch', type=int,
                            default=30,
                            help='max epoches during training')

        # hyper parameters
        parser.add_argument('--word_dropout', type=float,
                            default='0.04',
                            help='randomly set a token to be <UNK>')
        parser.add_argument('--dropout', type=float,
                            default='0.5',
                            help='the possiblity of dropout')
        parser.add_argument('--batch_size', type=int,
                            default=50,
                            help='batch size')
        parser.add_argument('--lr', type=float,
                            default=1.0,
                            help='learning rate')
        parser.add_argument('--max_len', type=int,
                            default=100,
                            help='max length of sentence')
        parser.add_argument('--att_len', type=int,
                            default=200,
                            help='the size of attention layer')
        parser.add_argument('--pos_dim', type=int,
                            default=30,
                            help='dimension of position embedding')

        # hyper parameters for rnn
        parser.add_argument('--hidden_size', type=int,
                            default=200,
                            help='the dimension of hidden units in RNN layer')
        parser.add_argument('--layers_num', type=int,
                            default=2,
                            help='num of RNN layers')

        # parser.add_argument('--L2_decay', type=float, default='1e-3',
        #                     help='L2 weight decay')

        args = parser.parse_args()
        return args

    def __set_seed(self, seed=1234):
        os.environ['PYTHONHASHSEED'] = '{}'.format(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)  # set seed for cpu
        torch.cuda.manual_seed(seed)  # set seed for current gpu
        torch.cuda.manual_seed_all(seed)  # set seed for all gpu

    def __config_backup(self, args):
        config_backup_path = os.path.join(self.model_dir, 'config.json')
        with open(config_backup_path, 'w', encoding='utf-8') as fw:
            json.dump(vars(args), fw, ensure_ascii=False)

    def print_config(self):
        for key in self.__dict__:
            print(key, end=' = ')
            print(self.__dict__[key])


if __name__ == '__main__':
    config = Config()
    config.print_config()
