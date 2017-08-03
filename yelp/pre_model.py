# -*- coding:utf8 -*-

from __future__ import print_function

from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Dense, LSTM

def load_model(model_path='./models/lstm_model.h5'):

    sequence_input = Input(shape=(400,), dtype='int32')
    embedded_sequences = Embedding(input_dim=102153, output_dim=300)(sequence_input)
    x = LSTM(300, dropout=0.2, recurrent_dropout=0.2, name='attention_layer')(embedded_sequences)
    preds = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=sequence_input, outputs=preds)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.load_weights(model_path)

    return model

def get_pre_model(model, save_path='./models/pre_model.h5'):
    pre_model = Sequential()
    for layer in model.layers[:-1]:
        print(layer)
        pre_model.add(layer)
    pre_model.save_weights(save_path)

if __name__=="__main__":
    model = load_model()
    get_pre_model(model)