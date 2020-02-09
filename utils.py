#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Version : Python 3.6


import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class WordEmbeddingLoader(object):
    """
    A loader for pre-trained word embedding
    """

    def __init__(self, config):
        self.path_word = config.embedding_path  # path of pre-trained word embedding
        self.word_dim = config.word_dim  # dimension of word embedding

    def load_word_vec(self):
        word2id = dict()  # word to wordID
        word_vec = list()  # wordID to word embedding

        word2id['PAD'] = len(word2id)  # PAD character
        word2id['UNK'] = len(word2id)  # words, out of vocabulary

        with open(self.path_word, 'r', encoding='utf-8') as fr:
            for line in fr:
                line = line.strip().split()
                if len(line) != self.word_dim + 1:
                    continue
                word2id[line[0]] = len(word2id)
                word_vec.append(np.asarray(line[1:], dtype=np.float32))

        word_vec = np.stack(word_vec).reshape(-1, self.word_dim)
        vec_mean, vec_std = word_vec.mean(), word_vec.std()
        extra_vec = np.random.normal(vec_mean, vec_std, size=(2, self.word_dim))

        special_char = ['SUBJ-ORGANIZATION', 'SUBJ-PERSON', 'OBJ-PERSON',
                        'OBJ-ORGANIZATION', 'OBJ-DATE', 'OBJ-NUMBER',
                        'OBJ-TITLE', 'OBJ-COUNTRY', 'OBJ-LOCATION',
                        'OBJ-CITY', 'OBJ-MISC', 'OBJ-STATE_OR_PROVINCE',
                        'OBJ-DURATION', 'OBJ-NATIONALITY', 'OBJ-CAUSE_OF_DEATH',
                        'OBJ-CRIMINAL_CHARGE', 'OBJ-RELIGION', 'OBJ-URL',
                        'OBJ-IDEOLOGY', 'Entity1', 'Entity2']
        for sc in special_char:
            word2id[sc] = len(word2id)
        special_emb = np.random.uniform(-1, 1, (len(special_char), self.word_dim))

        word_vec = np.concatenate((extra_vec, word_vec, special_emb), axis=0)
        # word_vec = np.concatenate((extra_vec, word_vec), axis=0)
        word_vec = word_vec.astype(np.float32).reshape(-1, self.word_dim)
        word_vec = torch.from_numpy(word_vec)
        return word2id, word_vec


class RelationLoader(object):
    """
    A loader for relation list
    """

    def __init__(self, config):
        self.data_dir = config.data_dir

        # returned parameters
        self.rel2id, self.id2rel, self.class_num = self.__load_relation()

    def __load_relation(self):
        relation_file = os.path.join(self.data_dir, 'relation2id.txt')
        rel2id = {}
        id2rel = {}
        with open(relation_file, 'r', encoding='utf-8') as fr:
            for line in fr:
                relation, id_s = line.strip().split()
                id_d = int(id_s)
                rel2id[relation] = id_d
                id2rel[id_d] = relation
        return rel2id, id2rel, len(rel2id)

    def get_relation(self):
        return self.rel2id, self.id2rel, self.class_num


class OneDataSet(Dataset):
    def __init__(self, rel2id, word2id, config):
        self.rel2id = rel2id
        self.word2id = word2id
        self.pos_dis = config.pos_dis
        self.max_len = config.max_len

    def __get_pos_index(self, x):
        if x < -self.pos_dis:
            return 0
        if x >= -self.pos_dis and x <= self.pos_dis:
            return x + self.pos_dis + 1
        if x > self.pos_dis:
            return 2 * self.pos_dis + 2

    def get_pos_index(self, x):
        return self.__get_pos_index(x)

    def __symbolize_sentence(self, e1, e2, sentence):
        """
            Args:
                sentence (list)

        """
        e1_pos = -1
        e2_pos = -1
        mask = []
        mask_flag = 1
        for i in range(len(sentence)):
            if e1_pos == -1 and sentence[i] == e1:
                e1_pos = i
                mask_flag += 1
            if e2_pos == -1 and sentence[i] == e2:
                e2_pos = i
                mask_flag += 1
            mask.append(mask_flag)
        words = []
        pos1 = []
        pos2 = []
        length = min(self.max_len, len(sentence))
        mask = mask[:length]
        for i in range(length):
            words.append(self.word2id.get(sentence[i].lower(), self.word2id['UNK']))
            pos1.append(self.__get_pos_index(i - e1_pos))
            pos2.append(self.__get_pos_index(i - e2_pos))
        if length < self.max_len:
            for i in range(length, self.max_len):
                mask.append(0)  # 'PAD' mask is zero
                words.append(self.word2id['PAD'])
                pos1.append(self.__get_pos_index(i - e1_pos))
                pos2.append(self.__get_pos_index(i - e2_pos))
        unit = np.asarray([words, pos1, pos2, mask], dtype=np.int64)
        unit = np.reshape(unit, newshape=(1, 4, self.max_len))
        return unit

    def symbolize_sentence(self, e1, e2, sentence):
        return self.__symbolize_sentence(e1, e2, sentence)


class TacredDataset(OneDataSet):
    def __init__(self, filename, rel2id, word2id, config):
        super().__init__(rel2id, word2id, config)
        # self.__get_pos_index = super().get_pos_index
        # self.__symbolize_sentence = super().symbolize_sentence
        self.filename = filename
        self.data_dir = config.data_dir
        self.dataset, self.label = self.__load_data()

    def __get_pos_index(self, x):
        # if x < -self.pos_dis:
        #     return 0
        # if x >= -self.pos_dis and x <= self.pos_dis:
        #     return x + self.pos_dis + 1
        # if x > self.pos_dis:
        #     return 2 * self.pos_dis + 2
        return x+self.max_len

    def __symbolize_sentence(self, e1_pos, e2_pos, sentence):
        """
            Args:
                sentence (list)

        """
        mask = [1] * len(sentence)
        if e1_pos[0] < e2_pos[0]:
            for i in range(e1_pos[0], e2_pos[1]+1):
                mask[i] = 2
            for i in range(e2_pos[1]+1, len(sentence)):
                mask[i] = 3
        else:
            for i in range(e2_pos[0], e1_pos[1]+1):
                mask[i] = 2
            for i in range(e1_pos[1]+1, len(sentence)):
                mask[i] = 3

        words = []
        pos1 = []
        pos2 = []
        length = min(self.max_len, len(sentence))
        mask = mask[:length]

        for i in range(length):
            words.append(self.word2id.get(sentence[i], self.word2id['UNK']))
            # pos1.append(self.__get_pos_index(i - e1_pos))
            # pos2.append(self.__get_pos_index(i - e2_pos))
            if i < e1_pos[0]:
                pos1.append(self.__get_pos_index(i-e1_pos[0]))
            elif i > e1_pos[1]:
                pos1.append(self.__get_pos_index(i-e1_pos[1]))
            else:
                pos1.append(self.__get_pos_index(0))

            if i < e2_pos[0]:
                pos2.append(self.__get_pos_index(i-e2_pos[0]))
            elif i > e2_pos[1]:
                pos2.append(self.__get_pos_index(i-e2_pos[1]))
            else:
                pos2.append(self.__get_pos_index(0))

        if length < self.max_len:
            for i in range(length, self.max_len):
                mask.append(0)  # 'PAD' mask is zero
                words.append(self.word2id['PAD'])
                # pos1.append(self.__get_pos_index(i - e1_pos))
                # pos2.append(self.__get_pos_index(i - e2_pos))
                if i < e1_pos[0]:
                    pos1.append(self.__get_pos_index(i-e1_pos[0]))
                elif i > e1_pos[1]:
                    pos1.append(self.__get_pos_index(i-e1_pos[1]))
                else:
                    pos1.append(self.__get_pos_index(0))

                if i < e2_pos[0]:
                    pos2.append(self.__get_pos_index(i-e2_pos[0]))
                elif i > e2_pos[1]:
                    pos2.append(self.__get_pos_index(i-e2_pos[1]))
                else:
                    pos2.append(self.__get_pos_index(0))

        unit = np.asarray([words, pos1, pos2, mask], dtype=np.int64)
        unit = np.reshape(unit, newshape=(1, 4, self.max_len))
        return unit

    def __load_data(self):
        path_data_file = os.path.join(self.data_dir, self.filename)
        data = []
        labels = []
        with open(path_data_file, 'r', encoding='utf-8') as fr:
            for line in fr:
                line = json.loads(line.strip())
                label = line['relation']
                sentence = line['token']
                e1_pos = (line['subj_start'], line['subj_end'])
                e2_pos = (line['obj_start'], line['obj_end'])
                label_idx = self.rel2id[label]

                ss, se = line['subj_start'], line['subj_end']  # 头实体span
                oss, oe = line['obj_start'], line['obj_end']  # 尾实体span
                sentence[ss:se+1] = ['SUBJ-'+line['subj_type']] * (se-ss+1)  # 替换头实体
                sentence[oss:oe+1] = ['OBJ-'+line['obj_type']] * (oe-oss+1)  # 替换尾实体
                # sentence[ss:se+1] = ['Entity1'] * (se-ss+1)  # 替换头实体
                # sentence[oss: oe+1] = ['Entity2'] * (oe-oss+1)  # 替换尾实体

                one_sentence = self.__symbolize_sentence(e1_pos, e2_pos, sentence)
                data.append(one_sentence)
                labels.append(label_idx)
        return data, labels

    def __getitem__(self, index):
        data = self.dataset[index]
        label = self.label[index]
        return data, label

    def __len__(self):
        return len(self.label)


class SemEvalDataset(OneDataSet):
    def __init__(self, filename, rel2id, word2id, config):
        super().__init__(rel2id, word2id, config)
        # self.__get_pos_index = super().get_pos_index
        # self.__symbolize_sentence = super().symbolize_sentence
        self.filename = filename
        self.data_dir = config.data_dir
        self.dataset, self.label = self.__load_data()

    def __get_pos_index(self, x):
        # if x < -self.pos_dis:
        #     return 0
        # if x >= -self.pos_dis and x <= self.pos_dis:
        #     return x + self.pos_dis + 1
        # if x > self.pos_dis:
        #     return 2 * self.pos_dis + 2
        return x+self.max_len

    def __symbolize_sentence(self, e1_pos, e2_pos, sentence):
        """
            Args:
                sentence (list)

        """
        mask = [1] * len(sentence)
        if e1_pos[0] < e2_pos[0]:
            for i in range(e1_pos[0], e2_pos[1]+1):
                mask[i] = 2
            for i in range(e2_pos[1]+1, len(sentence)):
                mask[i] = 3
        else:
            for i in range(e2_pos[0], e1_pos[1]+1):
                mask[i] = 2
            for i in range(e1_pos[1]+1, len(sentence)):
                mask[i] = 3

        words = []
        pos1 = []
        pos2 = []
        length = min(self.max_len, len(sentence))
        mask = mask[: length]

        for i in range(length):
            words.append(self.word2id.get(sentence[i], self.word2id['UNK']))
            # pos1.append(self.__get_pos_index(i - e1_pos))
            # pos2.append(self.__get_pos_index(i - e2_pos))
            if i < e1_pos[0]:
                pos1.append(self.__get_pos_index(i-e1_pos[0]))
            elif i > e1_pos[1]:
                pos1.append(self.__get_pos_index(i-e1_pos[1]))
            else:
                pos1.append(self.__get_pos_index(0))

            if i < e2_pos[0]:
                pos2.append(self.__get_pos_index(i-e2_pos[0]))
            elif i > e2_pos[1]:
                pos2.append(self.__get_pos_index(i-e2_pos[1]))
            else:
                pos2.append(self.__get_pos_index(0))

        if length < self.max_len:
            for i in range(length, self.max_len):
                mask.append(0)  # 'PAD' mask is zero
                words.append(self.word2id['PAD'])
                # pos1.append(self.__get_pos_index(i - e1_pos))
                # pos2.append(self.__get_pos_index(i - e2_pos))
                if i < e1_pos[0]:
                    pos1.append(self.__get_pos_index(i-e1_pos[0]))
                elif i > e1_pos[1]:
                    pos1.append(self.__get_pos_index(i-e1_pos[1]))
                else:
                    pos1.append(self.__get_pos_index(0))

                if i < e2_pos[0]:
                    pos2.append(self.__get_pos_index(i-e2_pos[0]))
                elif i > e2_pos[1]:
                    pos2.append(self.__get_pos_index(i-e2_pos[1]))
                else:
                    pos2.append(self.__get_pos_index(0))

        unit = np.asarray([words, pos1, pos2, mask], dtype=np.int64)
        unit = np.reshape(unit, newshape=(1, 4, self.max_len))
        return unit

    def __load_data(self):
        path_data_file = os.path.join(self.data_dir, self.filename)
        data = []
        labels = []
        with open(path_data_file, 'r', encoding='utf-8') as fr:
            lines = json.load(fr)
        for line in lines:
            label = line['relation']
            sentence = line['token']
            e1_pos = (line['subj_start'], line['subj_end'])
            e2_pos = (line['obj_start'], line['obj_end'])
            label_idx = self.rel2id[label]

            ss, se = line['subj_start'], line['subj_end']  # 头实体span
            oss, oe = line['obj_start'], line['obj_end']  # 尾实体span
            sentence[ss: se+1] = ['SUBJ-'+line['subj_type']] * (se-ss+1)  # 替换头实体
            sentence[oss: oe+1] = ['OBJ-'+line['obj_type']] * (oe-oss+1)  # 替换尾实体

            one_sentence = self.__symbolize_sentence(e1_pos, e2_pos, sentence)
            data.append(one_sentence)
            labels.append(label_idx)
        return data, labels

    def __getitem__(self, index):
        data = self.dataset[index]
        label = self.label[index]
        return data, label

    def __len__(self):
        return len(self.label)


class OneDataLoader(object):
    def __init__(self, rel2id, word2id, config):
        self.rel2id = rel2id
        self.word2id = word2id
        self.config = config

    def __collate_fn(self, batch):
        data, label = zip(*batch)  # unzip the batch data
        data = list(data)
        label = list(label)
        data = torch.from_numpy(np.concatenate(data, axis=0))
        label = torch.from_numpy(np.asarray(label, dtype=np.int64))
        return data, label

    def collate_fn(self, batch):
        return(self.__collate_fn(batch))


class TacredDataLoader(OneDataLoader):
    def __init__(self, rel2id, word2id, config):
        super().__init__(rel2id, word2id, config)
        self.__collate_fn = super().collate_fn

    def __get_dataset(self, filename, shuffle=False):
        dataset = TacredDataset(filename, self.rel2id, self.word2id, self.config)
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=2,
            collate_fn=self.__collate_fn
        )
        return loader

    def get_train(self):
        return self.__get_dataset('train.json', shuffle=True)

    def get_dev(self):
        return self.__get_dataset('dev.json', shuffle=False)

    def get_test(self):
        return self.__get_dataset('test.json', shuffle=False)


class SemEvalDataLoader(OneDataLoader):
    def __init__(self, rel2id, word2id, config):
        super().__init__(rel2id, word2id, config)
        self.__collate_fn = super().collate_fn

    def __get_dataset(self, filename, shuffle=False):
        dataset = SemEvalDataset(filename, self.rel2id, self.word2id, self.config)
        loader = DataLoader(
            dataset=dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=2,
            collate_fn=self.__collate_fn
        )
        return loader

    def get_train(self):
        return self.__get_dataset('train.json', shuffle=True)

    def get_dev(self):
        return self.__get_dataset('test.json', shuffle=False)

    def get_test(self):
        return self.__get_dataset('test.json', shuffle=False)


if __name__ == '__main__':
    from config import InitConfig
    config = InitConfig()
    word2id, word_vec = WordEmbeddingLoader(config).load_word_vec()
    rel2id, id2rel, class_num = RelationLoader(config).get_relation()
    loader = SemEvalDataLoader(rel2id, word2id, config)
    train_loader = loader.get_test()

    for step, (data, label) in enumerate(train_loader):
        print(type(data), data.shape)
        print(type(label), label.shape)
        break