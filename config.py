#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.6

import argparse
import torch
import os
import random
import json
import numpy as np

DATA_DIR = {
    "semeval": "/data6/htwang/resource/data/SemEval2010_in_TACRED_format",
    "tacred": "/data6/htwang/resource/data/TACRED",
    "nyt": "/data6/htwang/resource/data/NYT"
}
DATA_TYPE = {
    "semeval": "SemEval",
    "tacred": "TACRED",
    "nyt": "NYT"
}
EMBEDDING_DIR = {
    "glove": "/data6/htwang/resource/embedding/glove",
    "google": "/data6/htwang/resource/embedding/GoogleNews"
}
MODEL_NAME = {
    "cnn": "CNN",
    "pcnn": "PCNN",
    "crcnn": "CRCNN",
    "lstm": "LSTM",
    "bi-lstm": "BiLSTM",
    "att-blstm": "ATT-BLSTM",
    "pa-lstm": "PA-LSTM"
}


class Parameter(object):
    def __init__(self):
        pass

    @classmethod
    def __get_parameter(cls):
        parser = argparse.ArgumentParser()
        parser.description = 'config for models'

        # several key selective parameters
        parser.add_argument('--data_name', type=str,
                            default='tacred',
                            choices=['semeval', 'tacred', 'nyt'],
                            help='dataset for the model')
        parser.add_argument('--data_dir', type=str,
                            default=None,
                            help='dir to load data, dicided by `--data_name`')
        parser.add_argument('--output_dir', type=str,
                            default='./output',
                            help='dir to save output')
        parser.add_argument('--model_dir', type=str,
                            default=None,
                            help='dir to save models')

        # word embedding
        parser.add_argument('--embedding_type', type=str,
                            default='glove',
                            choices=['glove', 'google'],
                            help='embedding for word encode')
        parser.add_argument('--embedding_dir', type=str,
                            default=None,
                            help='pre_trained word embedding')
        parser.add_argument('--word_dim', type=int,
                            default=50,
                            help='dimension of word embedding')

        # train settings
        parser.add_argument('--model_name', type=str,
                            default='pa-lstm',
                            choices=['cnn', 'pcnn', 'crcnn',
                                     'lstm', 'bi-lstm', 'att-blstm', 'pa-lstm'],
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

        # hyper parameters
        parser.add_argument('--optim', type=str,
                            default='sgd',
                            choices=['adagrad', 'sgd', 'adadelta', 'adagrad'],
                            help='optimizer for the model optimization')
        parser.add_argument('--epoch', type=int,
                            default=30,
                            help='max epoches during training')
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
        parser.add_argument('--pos_dis', type=int,
                            default=50,
                            help='max distance of position embedding')
        parser.add_argument('--pos_dim', type=int,
                            default=30,
                            help='dimension of position embedding')

        # hyper parameters for cnn
        parser.add_argument('--filter_num', type=int,
                            default=256,
                            help='the number of filters in convolution')
        parser.add_argument('--window', type=int,
                            default='3',
                            help='the size of window in convolution')

        # hyper parameters for rnn
        parser.add_argument('--hidden_size', type=int,
                            default=200,
                            help='the dimension of hidden units in RNN layer')
        parser.add_argument('--layers_num', type=int,
                            default=2,
                            help='num of RNN layers')

        parser.add_argument('--L2_decay', type=float, default='1e-3',
                            help='L2 weight decay')

        args = parser.parse_args()
        return args

    @classmethod
    def get_args(cls):
        return cls.__get_parameter()


class InitConfig(object):
    """
    A class to initialize the program environment,
    including default parameters setting, random seed setting, check of file directory
    """

    def __init__(self):
        args = Parameter.get_args()
        for key in args.__dict__:
            setattr(self, key, args.__dict__[key])
        self.device = torch.device('cuda:{}'.format(self.cuda)
                                   if torch.cuda.is_available() else 'cpu')

        # data dir
        if self.data_dir is None:
            self.data_dir = DATA_DIR[self.data_name]

        # model dir
        if self.model_dir is None:
            self.model_dir = os.path.join(
                self.output_dir,
                '{}-{}'.format(DATA_TYPE[self.data_name],
                               MODEL_NAME[self.model_name])
            )

        # embedding
        if self.embedding_dir is None:
            self.embedding_dir = EMBEDDING_DIR[self.embedding_type]
        self.embedding_path = None
        if self.embedding_type == 'glove':
            self.embedding_path = os.path.join(
                self.embedding_dir,
                'glove.6B.{}d.txt'.format(self.word_dim))
        elif self.embedding_type == 'google':
            self.embedding_path = os.path.join(
                self.embedding_dir,
                'GoogleNews-vectors-negative300.txt')
        else:
            # waiting for updates
            pass

        self.model_path, self.result_path = self.__check_filedir(self.model_dir)
        self.__config_backup(args, self.model_dir)
        self.__set_seed(self.seed)

    def __set_seed(self, seed=0):
        """
        random seed setting
        """
        os.environ['PYTHONHASHSEED'] = '{}'.format(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def __check_filedir(self, model_dir):
        """
        check of file directory
        """
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_path = os.path.join(model_dir, 'checkpoints')
        result_path = os.path.join(model_dir, 'result')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        return model_path, result_path

    def __config_backup(self, args, model_dir):
        """
        config backup
        """
        config_backup_path = os.path.join(model_dir, 'config.json')
        with open(config_backup_path, 'w', encoding='utf-8') as fw:
            json.dump(args.__dict__, fw, ensure_ascii=False)

    def print_config(self):
        """
        print config
        """
        for key in self.__dict__:
            print(key, end=' = ')
            print(self.__dict__[key])


if __name__ == '__main__':
    config = InitConfig().print_config()
    # para=Parameter.get_args()
