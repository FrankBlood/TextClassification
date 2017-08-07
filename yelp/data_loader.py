# -*- coding:utf8 -*-

from __future__ import print_function
import os
import numpy as np
import pandas as pd
import json
import codecs
from config import Config


class Data_Loader(object):

    def __init__(self):
        print('Loading data...')

    def load_from_file(self, config):
        '''

        :param config:
        :return:
        '''

        data = []
        label = []

        print("Load data...")
        with codecs.open(config.data_path, encoding='utf8') as fp:
            for line in fp.readlines():
                json_data = json.loads(line.strip())
                text = json_data['text']
                stars = json_data['stars']
                data.append(text.strip())
                label.append([int(stars)-1])

        print("all data is", len(data))
        # print("all label is", set(label))
        return data, label

    def split(self, config, rate=0.2):
        '''

        :param config:
        :return:
        '''

        data = []
        label = []

        print("Load data...")
        with codecs.open(config.data_path, encoding='utf8') as fp:
            for line in fp.readlines():
                json_data = json.loads(line.strip())
                text = json_data['text']
                stars = json_data['stars']
                data.append(text.strip())
                label.append([int(stars)-1])

        print("all data is", len(data))
        # print("all label is", set(label))
        return data, label

    def stat(self, config):
        data, _ = self.load_from_file(config)
        line_len = []
        for line in data:
            line = line.strip().split()
            line_len.append(len(line))
        print(len(line_len))
        print(max(line_len))
        print(min(line_len))
        print(np.average(line_len))
        print(np.median(line_len))
        print(pd.value_counts(line_len, ascending=False))

def test_stat():
    config = Config()
    data_loader = Data_Loader()
    data_loader.stat(config)

def test_load():
    config = Config()
    data_loader = Data_Loader()
    data_loader.load_from_file(config)

if __name__ == '__main__':
    # test_load()
    test_stat()