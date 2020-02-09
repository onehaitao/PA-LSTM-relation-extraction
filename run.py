#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.6

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer
from torch.nn import init
from config import InitConfig
# from utils import WordEmbeddingLoader, RelationLoader
from utils import WordEmbeddingLoader, RelationLoader, SemEvalDataLoader, TacredDataLoader
from model import CNN, PCNN, BiLSTM, ATT_BLSTM, PA_LSTM
from evaluate import Eval


def change_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def train(model, criterion, loader, config):
    train_loader, dev_loader, _ = loader

    optimizer = None
    if config.optim == 'adam':
        # optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.L2_decay)
        optimizer = optim.Adam(model.parameters(), lr=config.lr)
    elif config.optim == 'sgd':
        # optimizer = optim.SGD(model.parameters(), lr=config.lr, weight_decay=config.L2_decay)
        optimizer = optim.SGD(model.parameters(), lr=config.lr)
    elif config.optim == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=config.lr, weight_decay=config.L2_decay)
    elif config.optim == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=config.lr, weight_decay=config.L2_decay,
                                  init_accu_value=0.1)
    else:
        pass

    print(model)
    print('traning model parameters:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print('%s :  %s' % (name, str(param.data.shape)))
    print('--------------------------------------')
    print('start to train the model ...')

    eval_tool = Eval(config)
    min_f1 = -float('inf')
    current_lr = config.lr
    for epoch in range(1, config.epoch+1):
        if epoch > 10:
            current_lr *= 0.9
            change_lr(optimizer, current_lr)
        model.train()
        for step, (data, label) in enumerate(train_loader):
            data = data.to(config.device)
            label = label.to(config.device)

            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        _, train_loss, _ = eval_tool.evaluate(model, criterion, train_loader)
        f1, dev_loss, _ = eval_tool.evaluate(model, criterion, dev_loader)

        f1_type = None
        if config.data_name == 'semeval':
            f1_type = 'macro_f1'
        elif config.data_name == 'tacred':
            f1_type = 'micro_f1'
        else:
            pass

        print('[%03d] train_loss: %.3f | dev_loss: %.3f | %s on dev: %.4f'
              % (epoch, train_loss, dev_loss, f1_type, f1), end=' ')
        if f1 > min_f1:
            min_f1 = f1
            # 直接保存模型会出现warning
            # /data6/htwang/anaconda3/envs/pytorch/lib/python3.6/site-packages/torch/serialization.py:292: UserWarning: Couldn't retrieve source code for container of type Embedding. It won't be checked for correctness upon loading.
            # "type " + obj.__name__ + ". It won't be checked "
            # torch.save(model, os.path.join(config.model_path, 'model.pkl'))
            torch.save(model.state_dict(), os.path.join(config.model_path, 'model.pkl'))
            print('>>> save models!')
        else:
            print()


def test(model, criterion, loader, config):
    print('--------------------------------------')
    print('start test ...')

    _, _, test_loader = loader
    model.load_state_dict(torch.load(os.path.join(config.model_path, 'model.pkl')))
    eval_tool = Eval(config)
    f1, test_loss, preds = eval_tool.evaluate(model, criterion, test_loader)

    f1_type = None
    if config.data_name == 'semeval':
        f1_type = 'macro_f1'
    elif config.data_name == 'tacred':
        f1_type = 'micro_f1'
    else:
        pass
    print('test_loss: %.3f | %s on test: %.2f%%' % (test_loss, f1_type, f1*100))
    return preds


if __name__ == '__main__':
    config = InitConfig()
    print('--------------------------------------')
    print('some config:')
    config.print_config()

    print('--------------------------------------')
    print('start to load data ...')
    word2id, word_vec = WordEmbeddingLoader(config).load_word_vec()
    rel2id, id2rel, class_num = RelationLoader(config).get_relation()

    loader = None
    if config.data_name == 'semeval':
        loader = SemEvalDataLoader(rel2id, word2id, config)
    elif config.data_name == 'tacred':
        loader = TacredDataLoader(rel2id, word2id, config)
    else:
        pass
    train_loader, dev_loader = None, None
    if config.mode == 1:  # train mode
        train_loader = loader.get_train()
        dev_loader = loader.get_dev()
    test_loader = loader.get_test()
    loader = [train_loader, dev_loader, test_loader]
    print('finish!')

    print('--------------------------------------')

    # if config.model_name == 'cnn':
    #     model = CNN(word_vec=word_vec, class_num=class_num, config=config)
    # elif config.model_name == 'pcnn':
    #     model = PCNN(word_vec=word_vec, class_num=class_num, config=config)
    # elif config.model_name == 'bi-lstm':
    #     model = BiLSTM(word_vec=word_vec, class_num=class_num, config=config)
    # elif config.model_name == 'att-blstm':
    #     model = ATT_BLSTM(word_vec=word_vec, class_num=class_num, config=config)
    # else:
    #     pass
    model = PA_LSTM(word_vec=word_vec, class_num=class_num, config=config)
    model = model.to(config.device)
    criterion = nn.CrossEntropyLoss()

    if config.mode == 1:  # train mode
        train(model, criterion, loader, config)
    preds = test(model, criterion, loader, config)

    # result_file = os.path.join(config.result_path, 'predicted_answer.txt')
    # if config.data_name == 'semeval':
    #     start = 8001
    #     with open(result_file, 'w', encoding='utf-8') as fw:
    #         for i in range(0, preds.shape[0]):
    #             fw.write(str(start + i) + '\t' + id2rel[int(preds[i])] + '\n')
    # elif config.data_name == 'tacred':
    #     with open(result_file, 'w', encoding='utf-8') as fw:
    #         for i in range(0, preds.shape[0]):
    #             fw.write(id2rel[int(preds[i])] + '\n')
    # else:
    #     pass


# class
class MyAdagrad(Optimizer):
    """My modification of the Adagrad optimizer that allows to specify an initial
    accumulater value. This mimics the behavior of the default Adagrad implementation
    in Tensorflow. The default PyTorch Adagrad uses 0 for initial acculmulator value.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        lr_decay (float, optional): learning rate decay (default: 0)
        init_accu_value (float, optional): initial accumulater value.
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(self, params, lr=1e-2, lr_decay=0, init_accu_value=0.1, weight_decay=0):
        defaults = dict(lr=lr, lr_decay=lr_decay, init_accu_value=init_accu_value,
                        weight_decay=weight_decay)
        super(MyAdagrad, self).__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['sum'] = torch.ones(p.data.size()).type_as(p.data) *\
                    init_accu_value

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['sum'].share_memory_()

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                state['step'] += 1

                if group['weight_decay'] != 0:
                    if p.grad.data.is_sparse:
                        raise RuntimeError("weight_decay option is not compatible with sparse gradients ")
                    grad = grad.add(group['weight_decay'], p.data)

                clr = group['lr'] / (1 + (state['step'] - 1) * group['lr_decay'])

                if p.grad.data.is_sparse:
                    grad = grad.coalesce()  # the update is non-linear so indices must be unique
                    grad_indices = grad._indices()
                    grad_values = grad._values()
                    size = torch.Size([x for x in grad.size()])

                    def make_sparse(values):
                        constructor = type(p.grad.data)
                        if grad_indices.dim() == 0 or values.dim() == 0:
                            return constructor()
                        return constructor(grad_indices, values, size)
                    state['sum'].add_(make_sparse(grad_values.pow(2)))
                    std = state['sum']._sparse_mask(grad)
                    std_values = std._values().sqrt_().add_(1e-10)
                    p.data.add_(-clr, make_sparse(grad_values / std_values))
                else:
                    state['sum'].addcmul_(1, grad, grad)
                    std = state['sum'].sqrt().add_(1e-10)
                    p.data.addcdiv_(-clr, grad, std)

        return loss
