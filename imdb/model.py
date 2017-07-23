# -*- coding:utf8 -*-

from __future__ import print_function
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Dense, LSTM, GRU, Conv1D, GlobalMaxPooling1D, MaxPooling1D
from keras.layers import TimeDistributed, RepeatVector, Permute, Lambda, Bidirectional, Dropout
from keras.layers.merge import concatenate, add, dot, multiply
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.layers import Activation
from keras.optimizers import RMSprop, Adam, SGD, Adagrad, Adadelta, Adamax, Nadam
from keras.layers.advanced_activations import PReLU

from config import Config

def cnn_based_rnn(config, embedding_matrix=None):
    """CNN based Attentive RNN Model

    :param config:
    :param embedding_matrix:
    :return: The model
    """

    print("Build CNN based Attentive RNN...")
    if embedding_matrix == None:
        # # embedding_matrix = np.zeros((config.max_features, config.embedding_dims))
        # numpy_rng = np.random.RandomState(4321)
        # embedding_matrix = numpy_rng.uniform(low=-0.05, high=0.05, size=(config.max_features, config.embedding_dims))
        embedding_layer = Embedding(config.max_features,
                                    config.embedding_dims,
                                    input_length=config.maxlen)

    else:
        embedding_layer = Embedding(config.max_features,
                                    config.embedding_dims,
                                    weights=[embedding_matrix],
                                    input_length=config.maxlen,
                                    trainable=False)

    rnn_layer = Bidirectional(GRU(config.lstm_output_size, dropout=config.dropout, recurrent_dropout=config.dropout))
    cnn_layer = Conv1D(activation="relu", padding="valid", strides=1, filters=config.nb_filter, kernel_size=config.filter_length)
    pooling_layer = GlobalMaxPooling1D()
    cnn_dense = Dense(config.hidden_dims)
    cnn_dropout1 = Dropout(config.dropout)
    cnn_dropout2 = Dropout(config.dropout)
    cnn_batchnormalization = BatchNormalization()
    cnn_dense1 = Dense(config.hidden_dims)

    sequence_input = Input(shape=(config.maxlen,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    cnn = cnn_layer(embedded_sequences)
    cnn = pooling_layer(cnn)
    cnn = cnn_dropout1(cnn)
    cnn = cnn_dense(cnn)
    cnn = cnn_dropout2(cnn)
    cnn = cnn_batchnormalization(cnn)

    cnn_t = cnn_dense1(cnn)

    a = multiply([cnn_t, embedded_sequences])

    a = Permute([2, 1])(a)

    a = Lambda(lambda x: K.sum(x, axis=1))(a)

    a = Activation('sigmoid')(a)

    embedded_sequences = Permute([2, 1])(embedded_sequences)

    x = multiply([a, embedded_sequences])

    x = Permute([2, 1])(x)

    x = rnn_layer(x)

    x = Dense(config.hidden_dims, activation='relu')(x)
    x = Dropout(config.dropout)(x)
    x = BatchNormalization()(x)

    preds = Dense(1, activation='sigmoid')(x)

    ########################################
    ## train the model
    ########################################
    model = Model(inputs=sequence_input, outputs=preds)
    model.compile(loss='binary_crossentropy',
                  optimizer='nadam',
                  metrics=['acc'])
    model.summary()
    return model

def bidirectional_lstm(config, embedding_matrix=None):
    """ Bidirectional LSTM model

    :param config:
    :return: the model
    """
    print('Build Bidirectional LSTM model...')

    if embedding_matrix == None:
        # # embedding_matrix = np.zeros((config.max_features, config.embedding_dims))
        # numpy_rng = np.random.RandomState(4321)
        # embedding_matrix = numpy_rng.uniform(low=-0.05, high=0.05, size=(config.max_features, config.embedding_dims))
        embedding_layer = Embedding(config.max_features,
                                    config.embedding_dims,
                                    input_length=config.maxlen)

    else:
        embedding_layer = Embedding(config.max_features,
                                    config.embedding_dims,
                                    weights=[embedding_matrix],
                                    input_length=config.maxlen,
                                    trainable=False)

    sequence_input = Input(shape=(config.maxlen,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    x = Bidirectional(LSTM(config.lstm_output_size))(embedded_sequences)
    x = Dropout(config.dropout)(x)
    preds = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=sequence_input, outputs=preds)

    # model = Sequential()
    # model.add(Embedding(config.max_features, config.embedding_dims, input_length=config.maxlen))
    # model.add(Bidirectional(LSTM(config.lstm_output_size)))
    # model.add(Dropout(config.dropout))
    # model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='nadam',
                  metrics=['acc'])
    model.summary()
    return model

def cnn_lstm(config, embedding_matrix=None):
    """CNN LSTM model

    :param config:
    :return: the model
    """

    print("Build CNN LSTM model...")

    if embedding_matrix == None:
        # # embedding_matrix = np.zeros((config.max_features, config.embedding_dims))
        # numpy_rng = np.random.RandomState(4321)
        # embedding_matrix = numpy_rng.uniform(low=-0.05, high=0.05, size=(config.max_features, config.embedding_dims))
        embedding_layer = Embedding(config.max_features,
                                    config.embedding_dims,
                                    input_length=config.maxlen)

    else:
        embedding_layer = Embedding(config.max_features,
                                    config.embedding_dims,
                                    weights=[embedding_matrix],
                                    input_length=config.maxlen,
                                    trainable=False)

    sequence_input = Input(shape=(config.maxlen,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Dropout(config.dropout)(embedded_sequences)
    x = Conv1D(nb_filter=config.nb_filter,
               filter_length=config.filter_length,
               border_mode='valid',
               activation='relu',
               subsample_length=1)(x)
    x = MaxPooling1D(pool_length=config.pool_length)(x)
    x = LSTM(config.lstm_output_size)(x)
    preds = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=sequence_input, outputs=preds)

    # model = Sequential()
    # model.add(Embedding(config.max_features, config.embedding_dims, input_length=config.maxlen))
    # model.add(Dropout(config.dropout))
    # model.add(Conv1D(nb_filter=config.nb_filter,
    #                  filter_length=config.filter_length,
    #                  border_mode='valid',
    #                  activation='relu',
    #                  subsample_length=1))
    # model.add(MaxPooling1D(pool_length=config.pool_length))
    # model.add(LSTM(config.lstm_output_size))
    # model.add(Dense(1))
    # model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    return model

def cnn(config, embedding_matrix=None):
    """CNN model

    :param config:
    :return: the model
    """
    print('Build CNN model...')

    if embedding_matrix == None:
        # # embedding_matrix = np.zeros((config.max_features, config.embedding_dims))
        # numpy_rng = np.random.RandomState(4321)
        # embedding_matrix = numpy_rng.uniform(low=-0.05, high=0.05, size=(config.max_features, config.embedding_dims))
        embedding_layer = Embedding(config.max_features,
                                    config.embedding_dims,
                                    input_length=config.maxlen)

    else:
        embedding_layer = Embedding(config.max_features,
                                    config.embedding_dims,
                                    weights=[embedding_matrix],
                                    input_length=config.maxlen,
                                    trainable=False)

    sequence_input = Input(shape=(config.maxlen,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    x = Conv1D(nb_filter=config.nb_filter,
               filter_length=config.filter_length,
               border_mode='valid',
               activation='relu',
               subsample_length=1)(embedded_sequences)
    x = GlobalMaxPooling1D()(x)
    x = Dense(config.hidden_dims, activation='relu')(x)
    x = Dropout(config.dropout)(x)
    preds = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=sequence_input, outputs=preds)

    # model = Sequential()
    #
    # # we start off with an efficient embedding layer which maps
    # # our vocab indices into embedding_dims dimensions
    # model.add(Embedding(config.max_features,
    #                     config.embedding_dims,
    #                     input_length=config.maxlen,
    #                     dropout=config.dropout))
    #
    # # we add a Convolution1D, which will learn nb_filter
    # # word group filters of size filter_length:
    # model.add(Conv1D(nb_filter=config.nb_filter,
    #                  filter_length=config.filter_length,
    #                  border_mode='valid',
    #                  activation='relu',
    #                  subsample_length=1))
    # # we use max pooling:
    # model.add(GlobalMaxPooling1D())
    #
    # # We add a vanilla hidden layer:
    # model.add(Dense(config.hidden_dims))
    # model.add(Dropout(config.dropout))
    # model.add(Activation('relu'))
    #
    # # We project onto a single unit output layer, and squash it with a sigmoid:
    # model.add(Dense(1))
    # model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='nadam',
                  metrics=['accuracy'])
    model.summary()
    return model

def lstm(config, embedding_matrix=None):
    """LSTM model

    :param config:
    :return: the model
    """
    print('Build LSTM model...')

    if embedding_matrix == None:
        # # embedding_matrix = np.zeros((config.max_features, config.embedding_dims))
        # numpy_rng = np.random.RandomState(4321)
        # embedding_matrix = numpy_rng.uniform(low=-0.05, high=0.05, size=(config.max_features, config.embedding_dims))
        embedding_layer = Embedding(config.max_features,
                                    config.embedding_dims,
                                    input_length=config.maxlen)

    else:
        embedding_layer = Embedding(config.max_features,
                                    config.embedding_dims,
                                    weights=[embedding_matrix],
                                    input_length=config.maxlen,
                                    trainable=False)

    sequence_input = Input(shape=(config.maxlen,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    x = LSTM(config.lstm_output_size, dropout=config.dropout, recurrent_dropout=config.dropout)(embedded_sequences)
    preds = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=sequence_input, outputs=preds)

    # model = Sequential()
    # model.add(Embedding(config.max_features, config.embedding_dims))
    # model.add(LSTM(config.lstm_output_size, dropout=config.dropout, recurrent_dropout=config.dropout))  # try using a GRU instead, for fun
    # model.add(Dense(1))
    # model.add(Activation('sigmoid'))

    print('Sucessfully built...')

    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    return model

if __name__ == '__main__':
    config = Config(max_feature=200000, maxlen=2570,
                    batch_size=50, embedding_dims=300, nb_filter=128,
                    filter_length=3, hidden_dims=300, nb_epoch=20, dropout=0.5,
                    pool_length=4, lstm_output_size=300, model_name='model',
                    embedding_file=None)
    # model = bidirectional_lstm(config)
    # model = cnn(config)
    # model = cnn_lstm(config)
    # model = lstm(config)
    model = cnn_based_rnn(config)