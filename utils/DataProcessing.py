import os
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import pandas as pd

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

class Corpus(object):

    def __init__(self,sen_len):
        self.dictionary = Dictionary()
        self.dictionary.add_word("UNK") # 要指定样本长度，不够的补0，默认0对应的单词为"UNK"
        self.texts,self.labels = self.tokenize(sen_len)

    def tokenize(self,sen_len):
        """
        得到字典，完成文本从单词到数字的转换
        :param sen_len: 文本长度
        :return:
        """
        df = pd.read_csv("./data/clean_review.csv")
        token_text = []
        tokens = 0
        labels = []
        for item in df.iterrows():
            line = item[1]["clean_review"] # 该行中，clean_review字段对应的值
            labels.append(int(item[1]["cat_id"])) # 该行中，cat_id字段对应的值，即这个样本的标签
            words = line.split(" ")

            tokens += len(words)
            for word in words:
                word = word.strip()
                if word:
                    self.dictionary.add_word(word)

            txt = torch.LongTensor(np.zeros(sen_len, dtype=np.int64)) # 构造长度为sen_len的tensor
            for index,word in enumerate(words[:sen_len]):
                word = word.strip()
                if word:
                    txt[index] = self.dictionary.word2idx[word]
            token_text.append(txt)

        return token_text,labels

class LSTMDataset(Dataset):

    def __init__(self,sen_len, corpus):
        corpus = corpus
        self.token_text = corpus.texts
        self.labels = corpus.labels
        self.sen_len = sen_len

    def __getitem__(self, index):
        """
        根据索引获取对应的特征和标签
        :param index:
        :return:
        """
        text = self.token_text[index]
        label = torch.LongTensor([self.labels[index]])
        return text, label

    def __len__(self):
        return len(self.labels)
