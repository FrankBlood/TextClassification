# -*- coding:utf8 -*-

from __future__ import print_function
import numpy as np
from keras.preprocessing import sequence
import pandas as pd
from keras.datasets import imdb
from config import Config
from data_loader import Data_Loader

class Pre_Process(object):

    def __init__(self):
        print('pre processing...')

    def process(self, config, data_loader):
        X_train, y_train, X_test, y_test = data_loader.load(config)
        print("Pad sequences (samples x time)")
        X_train = sequence.pad_sequences(X_train, maxlen=config.maxlen)
        X_test = sequence.pad_sequences(X_test, maxlen=config.maxlen)
        print('X_train shape:', X_train.shape)
        print('X_test shape:', X_test.shape)
        y_train = np.array(y_train)
        y_test = np.array(y_test)
        print('y_train shape:', y_train.shape)
        print('y_test shape:', y_test.shape)

        return  X_train, y_train, X_test, y_test

    def test(self):
        config = Config()
        data_loader = Data_Loader()
        X_train, y_train, X_test, y_test = self.process(config, data_loader)
        print(X_train[0])
        print(y_train[0])

if __name__ == "__main__":
    pre_process = Pre_Process()
    pre_process.test()