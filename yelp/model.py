# -*- coding:utf8 -*-

from __future__ import print_function
import numpy as np

from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Dense, LSTM, GRU, Conv1D, GlobalMaxPooling1D, MaxPooling1D, GlobalAveragePooling1D
from keras.layers import TimeDistributed, RepeatVector, Permute, Lambda, Bidirectional, Dropout
from keras.layers.merge import concatenate, add, dot, multiply
from keras.layers.normalization import BatchNormalization
from keras import backend as K
from keras.layers import Activation
from recurrent import ATTENTION_INNER_GRU
from keras.optimizers import RMSprop, Adam, SGD, Adagrad, Adadelta, Adamax, Nadam
from keras.layers.advanced_activations import PReLU

from config import Config


def pre_attention_inner_rnn(config, embedding_matrix=None):
    """Pre Attention INNER RNN

    :param config:
    :param embedding_matrix:
    :return: The model
    """

    print("Build Pre Attention INNER RNN...")

    if embedding_matrix == None:
        # Zero init...
        # embedding_matrix = np.zeros((config.max_features, config.embedding_dims))
        # Random init...
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

    ########################################
    ## Input and Embedding
    ########################################
    sequence_input = Input(shape=(config.maxlen,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    ########################################
    ## Pre Attention -- LSTM
    ########################################
    attention = LSTM(300, dropout=0.2, recurrent_dropout=0.2, name='attention_layer', trainable=False)(embedded_sequences)
    attention = Dense(300, activation='relu')(attention)
    # rnn_model = Model(inputs=sequence_input, outputs=rnn)
    #
    # rnn_model.load_weights('./models/pre_model.h5')
    #
    # attention = rnn_model.predict_on_batch(sequence_input)

    ########################################
    ## Attention INNER GRU Model
    ########################################
    x = Bidirectional(ATTENTION_INNER_GRU(config.lstm_output_size,
                                          attention=attention,
                                          dropout=config.dropout,
                                          recurrent_dropout=config.dropout))(embedded_sequences)

    x = Dense(config.hidden_dims, activation='relu')(x)
    x = Dropout(config.dropout)(x)
    x = BatchNormalization()(x)

    preds = Dense(5, activation='softmax')(x)

    ########################################
    ## train the model
    ########################################
    model = Model(inputs=sequence_input, outputs=preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='nadam',
                  metrics=['acc'])
    model.summary()
    return model


def rnn_cat_rnn(config, embedding_matrix=None):
    """RNN Cat Embedding GRU Model

    :param config:
    :param embedding_matrix:
    :return: The model
    """

    print("Build RNN CAT RNN...")
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

    ########################################
    ## Input and Embedding
    ########################################
    sequence_input = Input(shape=(config.maxlen,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    ########################################
    ## ATTENTION: RNN Model -- also GRU
    ########################################
    attention = GRU(config.lstm_output_size,
                    return_sequences=False,
                    dropout=config.dropout,
                    recurrent_dropout=config.dropout)(embedded_sequences)

    # attention = Dense(300, activation='relu')(attention)

    attention = RepeatVector(config.maxlen)(attention)

    # x = concatenate([embedded_sequences, attention])
    x = multiply([embedded_sequences, attention])

    ########################################
    ## Common RNN Model -- GRU
    ########################################
    x = Bidirectional(GRU(config.lstm_output_size,
                          dropout=config.dropout,
                          recurrent_dropout=config.dropout))(x)

    x = Dense(config.hidden_dims, activation='relu')(x)
    x = Dropout(config.dropout)(x)
    x = BatchNormalization()(x)

    preds = Dense(5, activation='softmax')(x)

    ########################################
    ## train the model
    ########################################
    model = Model(inputs=sequence_input, outputs=preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='nadam',
                  metrics=['acc'])
    model.summary()
    return model


def rnn_inner_rnn(config, embedding_matrix=None):
    """CNN add Attentive RNN Model

    :param config:
    :param embedding_matrix:
    :return: The model
    """

    print("Build RNN Attentive INNER RNN...")
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

    rnn = GRU(config.lstm_output_size,
              dropout=config.dropout,
              recurrent_dropout=config.dropout)(embedded_sequences)

    x = Bidirectional(ATTENTION_INNER_GRU(config.lstm_output_size,
                                          attention=rnn,
                                          dropout=config.dropout,
                                          recurrent_dropout=config.dropout))(embedded_sequences)

    x = Dense(config.hidden_dims, activation='relu')(x)
    x = Dropout(config.dropout)(x)
    x = BatchNormalization()(x)

    preds = Dense(5, activation='softmax')(x)

    ########################################
    ## train the model
    ########################################
    model = Model(inputs=sequence_input, outputs=preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='nadam',
                  metrics=['acc'])
    model.summary()
    return model


def cnn_inner_rnn(config, embedding_matrix=None):
    """CNN Attentive INNER RNN Model

    :param config:
    :param embedding_matrix:
    :return: cnn inner rnn model
    """

    print("Build CNN Attentive INNER RNN Model...")
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

    cnn_layer = Conv1D(activation="relu", padding="valid", strides=1,
                       filters=config.nb_filter, kernel_size=config.filter_length)
    pooling_layer = GlobalMaxPooling1D()
    cnn_dense = Dense(config.hidden_dims, activation='relu')
    cnn_dropout1 = Dropout(config.dropout)
    cnn_dropout2 = Dropout(config.dropout)
    cnn_batchnormalization = BatchNormalization()

    sequence_input = Input(shape=(config.maxlen,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    cnn = cnn_layer(embedded_sequences)
    cnn = pooling_layer(cnn)
    cnn = cnn_dropout1(cnn)
    cnn = cnn_dense(cnn)
    cnn = cnn_dropout2(cnn)
    cnn = cnn_batchnormalization(cnn)

    x = Bidirectional(ATTENTION_INNER_GRU(config.lstm_output_size,
                                          attention=cnn,
                                          dropout=config.dropout,
                                          recurrent_dropout=config.dropout))(embedded_sequences)

    x = Dense(config.hidden_dims, activation='relu')(x)
    x = Dropout(config.dropout)(x)
    x = BatchNormalization()(x)

    preds = Dense(5, activation='softmax')(x)

    ########################################
    ## train the model
    ########################################
    model = Model(inputs=sequence_input, outputs=preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='nadam',
                  metrics=['acc'])
    model.summary()
    return model


def cnn_add_rnn(config, embedding_matrix=None):
    """CNN add Attentive RNN Model

    :param config:
    :param embedding_matrix:
    :return: The model
    """

    print("Build CNN add Attentive RNN...")
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

    rnn_layer = Bidirectional(LSTM(config.lstm_output_size, dropout=config.dropout, recurrent_dropout=config.dropout, return_sequences=True))
    cnn_layer = Conv1D(activation="relu", padding="valid", strides=1, filters=config.nb_filter, kernel_size=config.filter_length)
    pooling_layer = GlobalMaxPooling1D()
    cnn_dense = Dense(config.hidden_dims, activation='relu')
    cnn_dropout1 = Dropout(config.dropout)
    cnn_dropout2 = Dropout(config.dropout)
    cnn_batchnormalization = BatchNormalization()
    cnn_dense1 = Dense(config.hidden_dims, activation='tanh')
    cnn_dense2 = Dense(config.hidden_dims*2, activation='tanh')

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

    cnn_t2 = cnn_dense2(cnn)
    a2 = multiply([cnn_t2, x])
    a2 = Permute([2, 1])(a2)
    a2 = Lambda(lambda x: K.sum(x, axis=1))(a2)
    a2 = Activation('softmax')(a2)
    x = Permute([2, 1])(x)
    x = multiply([a2, x])
    x = Permute([2, 1])(x)
    x = Lambda(lambda x: K.sum(x, axis=1))(x)

    x = Dense(config.hidden_dims, activation='relu')(x)
    x = Dropout(config.dropout)(x)
    x = BatchNormalization()(x)

    preds = Dense(5, activation='softmax')(x)

    ########################################
    ## train the model
    ########################################
    model = Model(inputs=sequence_input, outputs=preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='nadam',
                  metrics=['acc'])
    model.summary()
    # plot_model(model, to_file=config.model_name+'.png')
    return model


def complex_cnn_based_rnn(config, embedding_matrix=None):
    '''Complex CNN based RNN

    :param config:
    :param embedding_matrix:
    :return:
    '''
    print("Build Complex CNN based Attentive RNN...")
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

    rnn_layer = Bidirectional(GRU(config.lstm_output_size,
                                  dropout=config.dropout,
                                  recurrent_dropout=config.dropout))

    # cnn_layer = Conv1D(filters=config.nb_filter,
    #                    kernel_size=config.filter_length, padding = "valid", activation="relu", strides=1)
    #
    # conv1 = Conv1D(filters=config.nb_filter,
    #                kernel_size=1, padding="valid", strides=1, activation='relu')

    conv2 = Conv1D(filters=config.nb_filter,
                   kernel_size=2, padding="valid", strides=1, activation='relu')

    conv3 = Conv1D(filters=config.nb_filter,
                   kernel_size=3, padding="valid", strides=1, activation='relu')

    conv4 = Conv1D(filters=config.nb_filter,
                   kernel_size=4, padding="valid", strides=1, activation='relu')

    # conv5 = Conv1D(filters=config.nb_filter,
    #                kernel_size=5, padding='same', activation='relu')
    #
    # conv6 = Conv1D(filters=config.nb_filter,
    #                kernel_size=6, padding='same', activation='relu')
    #
    # pooling_layer = GlobalMaxPooling1D()
    cnn_dense = Dense(config.hidden_dims, activation='relu')
    # cnn_dropout1 = Dropout(0.2)
    # cnn_dropout2 = Dropout(0.2)
    # cnn_batchnormalization = BatchNormalization()
    # cnn_repeatvector = RepeatVector(config.embedding_dims)
    # cnn_dense1 = Dense(300, activation='relu')

    sequence_input = Input(shape=(config.maxlen,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    # conv1a = conv1(embedded_sequences)
    # glob1a = GlobalAveragePooling1D()(conv1a)
    # glob1a = Dropout(config.dropout)(glob1a)
    # glob1a = BatchNormalization()(glob1a)

    conv2a = conv2(embedded_sequences)
    glob2a = GlobalAveragePooling1D()(conv2a)
    glob2a = Dropout(config.dropout)(glob2a)
    glob2a = BatchNormalization()(glob2a)

    conv3a = conv3(embedded_sequences)
    glob3a = GlobalAveragePooling1D()(conv3a)
    glob3a = Dropout(config.dropout)(glob3a)
    glob3a = BatchNormalization()(glob3a)

    conv4a = conv4(embedded_sequences)
    glob4a = GlobalAveragePooling1D()(conv4a)
    glob4a = Dropout(config.dropout)(glob4a)
    glob4a = BatchNormalization()(glob4a)

    # conv5a = conv5(embedded_sequences)
    # glob5a = GlobalAveragePooling1D()(conv5a)
    # glob5a = Dropout(config.dropout)(glob5a)
    # glob5a = BatchNormalization()(glob5a)
    #
    # conv6a = conv6(embedded_sequences)
    # glob6a = GlobalAveragePooling1D()(conv6a)
    # glob6a = Dropout(config.dropout)(glob6a)
    # glob6a = BatchNormalization()(glob6a)

    cnn = concatenate([glob2a, glob3a, glob4a])

    # print(np.shape(cnn))
    # print(cnn.shape)

    cnn_t = cnn_dense(cnn)

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

    preds = Dense(5, activation='softmax')(x)

    ########################################
    ## train the model
    ########################################
    model = Model(inputs=sequence_input, outputs=preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='nadam',
                  metrics=['acc'])
    model.summary()
    return model


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

    ########################################
    ## All Used Layers
    ########################################
    rnn_layer = Bidirectional(GRU(config.lstm_output_size, dropout=config.dropout, recurrent_dropout=config.dropout))
    cnn_layer = Conv1D(activation="relu", padding="valid", strides=1, filters=config.nb_filter, kernel_size=config.filter_length)
    pooling_layer = GlobalMaxPooling1D()
    cnn_dense = Dense(config.hidden_dims, activation='relu')
    cnn_dropout1 = Dropout(config.dropout)
    cnn_dropout2 = Dropout(config.dropout)
    cnn_batchnormalization = BatchNormalization()
    cnn_dense1 = Dense(config.hidden_dims)

    ########################################
    ## Input and Embedding
    ########################################
    sequence_input = Input(shape=(config.maxlen,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    ########################################
    ## Attention Layer: CNN
    ########################################
    cnn = cnn_layer(embedded_sequences)
    cnn = pooling_layer(cnn)
    cnn = cnn_dropout1(cnn)
    cnn = cnn_dense(cnn)
    cnn = cnn_dropout2(cnn)
    cnn = cnn_batchnormalization(cnn)

    ########################################
    ## Attention Action
    ########################################
    cnn_t = cnn_dense1(cnn)
    a = multiply([cnn_t, embedded_sequences])
    a = Permute([2, 1])(a)
    a = Lambda(lambda x: K.sum(x, axis=1))(a)
    a = Activation('sigmoid')(a)
    embedded_sequences = Permute([2, 1])(embedded_sequences)
    x = multiply([a, embedded_sequences])
    x = Permute([2, 1])(x)

    ########################################
    ## Output Layers
    ########################################
    x = rnn_layer(x)
    x = Dense(config.hidden_dims, activation='relu')(x)
    x = Dropout(config.dropout)(x)
    x = BatchNormalization()(x)

    preds = Dense(5, activation='softmax')(x)

    ########################################
    ## train the model
    ########################################
    model = Model(inputs=sequence_input, outputs=preds)
    model.compile(loss='categorical_crossentropy',
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
    preds = Dense(5, activation='softmax')(x)
    model = Model(inputs=sequence_input, outputs=preds)

    # model = Sequential()
    # model.add(Embedding(config.max_features, config.embedding_dims, input_length=config.maxlen))
    # model.add(Bidirectional(LSTM(config.lstm_output_size)))
    # model.add(Dropout(config.dropout))
    # model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='categorical_crossentropy',
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
    preds = Dense(5, activation='softmax')(x)
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

    model.compile(loss='categorical_crossentropy',
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
    preds = Dense(5, activation='softmax')(x)

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

    model.compile(loss='categorical_crossentropy',
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
    preds = Dense(5, activation='softmax')(x)

    model = Model(inputs=sequence_input, outputs=preds)

    # model = Sequential()
    # model.add(Embedding(config.max_features, config.embedding_dims))
    # model.add(LSTM(config.lstm_output_size, dropout=config.dropout, recurrent_dropout=config.dropout))  # try using a GRU instead, for fun
    # model.add(Dense(1))
    # model.add(Activation('sigmoid'))

    print('Sucessfully built...')

    # try using different optimizers and different optimizer configs
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    return model


if __name__ == '__main__':
    config = Config(max_feature=102153, maxlen=2570,
                    batch_size=50, embedding_dims=300, nb_filter=128,
                    filter_length=3, hidden_dims=300, nb_epoch=20, dropout=0.5,
                    pool_length=4, lstm_output_size=300, model_name='model',
                    embedding_file=None)
    # model = bidirectional_lstm(config)
    # model = cnn(config)
    # model = cnn_lstm(config)
    # model = lstm(config)
    model = cnn_based_rnn(config)
    # model = cnn_add_rnn(config)
    # model = complex_cnn_based_rnn(config)
    # model = cnn_inner_rnn(config)
    # model = rnn_inner_rnn(config)
    # model = rnn_cat_rnn(config)
    # model = pre_attention_inner_rnn(config)