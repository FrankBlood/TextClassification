# -*- coding:utf8 -*-

from __future__ import print_function
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Dropout, Bidirectional
from keras.layers import Activation
from keras.layers import Input, Embedding, Dense, LSTM, Convolution1D, GlobalMaxPooling1D, MaxPooling1D

from config import Config

def bidirectional_lstm(config):
    """ Bidirectional LSTM model

    :param config:
    :return: the model
    """
    print('Build Bidirectional LSTM model...')
    model = Sequential()
    model.add(Embedding(config.max_features, config.embedding_dims, input_length=config.maxlen))
    model.add(Bidirectional(LSTM(config.lstm_output_size)))
    model.add(Dropout(config.dropout))
    model.add(Dense(1, activation='sigmoid'))
    model.compile('nadam', 'binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

def cnn_lstm(config):
    """CNN LSTM model

    :param config:
    :return: the model
    """

    print("Build CNN LSTM model...")
    model = Sequential()
    model.add(Embedding(config.max_features, config.embedding_dims, input_length=config.maxlen))
    model.add(Dropout(config.dropout))
    model.add(Convolution1D(nb_filter=config.nb_filter,
                            filter_length=config.filter_length,
                            border_mode='valid',
                            activation='relu',
                            subsample_length=1))
    model.add(MaxPooling1D(pool_length=config.pool_length))
    model.add(LSTM(config.lstm_output_size))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    return model

def cnn(config):
    """CNN model

    :param config:
    :return: the model
    """
    print('Build CNN model...')
    model = Sequential()

    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    model.add(Embedding(config.max_features,
                        config.embedding_dims,
                        input_length=config.maxlen,
                        dropout=config.dropout))

    # we add a Convolution1D, which will learn nb_filter
    # word group filters of size filter_length:
    model.add(Convolution1D(nb_filter=config.nb_filter,
                            filter_length=config.filter_length,
                            border_mode='valid',
                            activation='relu',
                            subsample_length=1))
    # we use max pooling:
    model.add(GlobalMaxPooling1D())

    # We add a vanilla hidden layer:
    model.add(Dense(config.hidden_dims))
    model.add(Dropout(config.dropout))
    model.add(Activation('relu'))

    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='nadam',
                  metrics=['accuracy'])
    model.summary()
    return model

def lstm(config):
    """LSTM model

    :param config:
    :return: the model
    """
    print('Build LSTM model...')
    model = Sequential()
    model.add(Embedding(config.max_features, config.embedding_dims))
    model.add(LSTM(config.hidden_dims, dropout=config.dropout, recurrent_dropout=config.dropout))  # try using a GRU instead, for fun
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    return model

if __name__ == '__main__':
    config = Config()
    # model = bidirectional_lstm(config)
    # model = cnn(config)
    # model = cnn_lstm(config)
    model = lstm(config)