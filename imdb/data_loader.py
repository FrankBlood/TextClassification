# -*- coding:utf8 -*-

from __future__ import print_function
import numpy as np
import pandas as pd
from keras.datasets import imdb
from config import Config


class Data_Loader(object):

    def __init__(self):
        print('Loading data...')

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


    def test(self):
        config = Config()
        X_train, y_train, X_test, y_test = self.load(config)
        print(X_train[0])
        print(y_train[0])

if __name__ == '__main__':
    data_loader = Data_Loader()
    data_loader.stat()