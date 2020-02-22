#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.6

import os
import json


relation = set()


def convert_from_json_file(path_src, path_des):
    with open(path_src, 'r', encoding='utf-8') as fr:
        info = json.load(fr)

    with open(path_des, 'w', encoding='utf-8') as fw:
        for subinfo in info:
            json.dump(subinfo, fw, ensure_ascii=False)
            fw.write('\n')
            rel = subinfo['relation']
            relation.add(rel)


def generate_relaiton2id(path_des):
    relation_list = sorted(relation, reverse=False)
    with open(path_des, 'w', encoding='utf-8') as fw:
        for id_idx, rel in enumerate(relation_list):
            fw.write('%s\t%d\n' % (rel, id_idx))


if __name__ == '__main__':
    path_src = './tacred_LDC2018T24/tacred/data/json'
    path_des = './'
    dataset = ['train.json', 'dev.json', 'test.json']
    for subset in dataset:
        convert_from_json_file(
            os.path.join(path_src, subset),
            os.path.join(path_des, subset)
        )
    generate_relaiton2id(os.path.join(path_des, 'relation2id.txt'))
