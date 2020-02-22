#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.6

import numpy as np
import torch


def tacred_scorer(predict_label, true_label):
    """
    scorer for TACRED dataset, excluding no_relation (NA)
    Args:
        ** predict_label **: (numpy.arrary [n]) predicted results
        ** true_label ** : (numpy.arrary [n]) ground truth
    Return:
        ** micro_f1 **: (float) micro F1-score:
    """
    assert true_label.shape[0] == predict_label.shape[0]
    from collections import Counter
    correct_excluding_na = Counter()
    pred_excluding_na = Counter()
    truth_excluding_na = Counter()
    for i in range(true_label.shape[0]):
        pred = int(predict_label[i])
        truth = int(true_label[i])
        if pred == 0 and truth == 0:
            pass
        elif pred == 0 and truth != 0:
            truth_excluding_na[truth] += 1
        elif pred != 0 and truth == 0:
            pred_excluding_na[pred] += 1
        elif pred != 0 and truth != 0:
            pred_excluding_na[pred] += 1
            truth_excluding_na[truth] += 1
            if pred == truth:
                correct_excluding_na[pred] += 1
    try:
        micro_precision = float(sum(correct_excluding_na.values())) / \
            float(sum(pred_excluding_na.values()))
    except:
        micro_precision = 1.0

    try:
        micro_recall = float(sum(correct_excluding_na.values())) / \
            float(sum(truth_excluding_na.values()))
    except:
        micro_recall = 0.0
    try:
        micro_f1 = 2.0 * micro_precision * \
            micro_recall / (micro_precision + micro_recall)
    except:
        micro_f1 = 0.0
    return micro_f1


class Eval(object):
    def __init__(self, config):
        self.device = config.device

    def evaluate(self, model, criterion, data_loader):
        predict_label = []
        true_label = []
        total_loss = 0.0
        with torch.no_grad():
            model.eval()
            for _, (data, label) in enumerate(data_loader):
                data = data.to(self.device)
                label = label.to(self.device)

                logits = model(data)
                loss = criterion(logits, label)
                total_loss += loss.item() * logits.shape[0]

                _, pred = torch.max(logits, dim=1)  # replace softmax with max function, same impacts
                pred = pred.cpu().detach().numpy().reshape((-1, 1))
                label = label.cpu().detach().numpy().reshape((-1, 1))
                predict_label.append(pred)
                true_label.append(label)
        predict_label = np.concatenate(predict_label, axis=0).reshape(-1).astype(np.int64)
        true_label = np.concatenate(true_label, axis=0).reshape(-1).astype(np.int64)
        eval_loss = total_loss / predict_label.shape[0]

        f1 = tacred_scorer(predict_label, true_label)
        return f1, eval_loss
