# -*- coding:utf8 -*-

from __future__ import print_function
import os
import numpy as np
import pandas as pd
import codecs
from keras.datasets import imdb
from config import Config


class Data_Loader(object):

    def __init__(self):
        print('Loading data...')

    def load_from_file(self, config, mode='train'):

        if mode == 'train':
            path = config.train_path
        else:
            path = config.test_path

        neg_path = path + 'neg/'
        pos_path = path + 'pos/'

        neg_list = os.listdir(neg_path)
        pos_list = os.listdir(pos_path)

        data = []
        label = []

        print("Load neg...")
        for neg in neg_list:
            with codecs.open(neg_path + neg) as fp:
                text = fp.read()
                data.append(text.strip())
                label.append(0)

        print("neg data is", len(data))

        print("Loading pos...")
        for pos in pos_list:
            with codecs.open(pos_path + pos) as fp:
                text = fp.read()
                data.append(text.strip())
                label.append(1)

        print("all data is", len(data))

        return data, label

    def load(self, config):
        """Load data

        :param config:
        :return: the data of imdb
        """
        (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=config.max_features)
        print(len(X_train), 'train sequences')
        print(len(X_test), 'test sequences')
        print('x_train set\' shape', np.shape(X_train))
        print('y_train set\' shape', np.shape(y_train))
        print('x_test set\' shape', np.shape(X_test))
        print('y_test set\' shape', np.shape(y_test))

        return X_train, y_train, X_test, y_test

    def stat(self):
        """Statistic data

        :return: None
        """
        print('statistic...')
        config = Config()
        X_train, y_train, X_test, y_test = self.load(config)
        len_num = []
        for X in X_train+X_test:
            len_num.append(len(X))
        print('max len num is', max(len_num))
        len_num = pd.Series(len_num)
        print('len num count is\n', len_num.value_counts())

    def show_example(self):
        print('show example...')
        config = Config(word_vocb_path='./data/imdb.vocab')
        index_word = config.get_index_word()
        X_train, _, _, _ = self.load(config)
        print(X_train[0])
        new_text = []
        for index in X_train[0]:
            new_text.append(index_word[index].strip())
        print(' '.join(new_text))

    def test(self):
        config = Config()
        X_train, y_train, X_test, y_test = self.load(config)
        print(X_train[0])
        print(y_train[0])

if __name__ == '__main__':
    data_loader = Data_Loader()
    data_loader.show_example()
    # data_loader.stat()