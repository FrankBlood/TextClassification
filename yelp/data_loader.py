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
                label.append(stars)

        print("all data is", len(data))
        print("all label is", set(label))
        return data, label

def test_load():
    config = Config()
    data_loader = Data_Loader()
    data_loader.load_from_file(config)

if __name__ == '__main__':
    test_load()
