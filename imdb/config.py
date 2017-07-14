# -*- coding:utf8 -*-

from __future__ import print_function
import codecs
import numpy as np

class Config(object):

    def __init__(self, max_feature=200000, maxlen=2570,
                 batch_size=50, embedding_dims=50, nb_filter=250,
                 filter_length=3, hidden_dims=250, nb_epoch=2, dropout=0.2,
                 pool_length=4, lstm_output_size=70, model_name='model',
                 embedding_file=None, word_vocb_path=None,
                 trian_path=None, test_path=None):

        print("config setting...")
        self.max_features = max_feature
        self.maxlen = maxlen
        self.batch_size = batch_size
        self.embedding_dims = embedding_dims
        self.nb_filter = nb_filter
        self.filter_length = filter_length
        self.hidden_dims = hidden_dims
        self.nb_epoch = nb_epoch
        self.dropout = dropout
        self.pool_length = pool_length
        self.lstm_output_size = lstm_output_size
        self.model_name = model_name
        self.embedding_file = embedding_file
        self.word_vocb_path = word_vocb_path
        self.train_path = trian_path
        self.test_path = test_path

    def get_word_index(self):
        print('Preparing word index...')
        word_index = {}
        count = 1
        with codecs.open(self.word_vocb_path) as fp:
            for word in fp.readlines():
                word_index[word] = count
                count += 1
        return word_index

    def get_index_word(self):
        print('Preparing index word...')
        index_word = {}
        count = 1
        with codecs.open(self.word_vocb_path) as fp:
            for word in fp.readlines():
                index_word[count] = word
                count += 1
        return index_word

    def get_embedding_matrix(self, word_index):
        print('Preparing embedding matrix')

        nb_words = min(self.max_features, len(word_index)) + 1

        # embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
        numpy_rng = np.random.RandomState(4321)
        embedding_matrix = numpy_rng.uniform(low=-0.05, high=0.05, size=(nb_words, self.embedding_dims))
        embeddings_from_file = {}
        with codecs.open(self.embedding_file) as embedding_file:
            for line in embedding_file.readlines():
                fields = line.strip().split(' ')
                word = fields[0]
                vector = np.array(fields[1:], dtype='float32')
                embeddings_from_file[word] = vector

        count = 0
        for word, i in word_index.items():
            if word in embeddings_from_file:
                embedding_matrix[i] = embeddings_from_file[word]
                count += 1
        print('Null word embeddings: %d' % (nb_words - count))

        return embedding_matrix

    def set_from_json(self, json_data):
        # TODO: 设置从json数据中读取设置
        pass

    def print_config(self):
        print('max_features', self.max_features)
        print('maxlen', self.maxlen)
        print('batch_size', self.batch_size)
        print('embedding_dims', self.embedding_dims)
        print('nb_filter', self.nb_filter)
        print('filter_length', self.filter_length)
        print('hidden_dims', self.hidden_dims)
        print('nb_epoch', self.nb_epoch)
        print('dropout', self.dropout)
        print('pool_length', self.pool_length)
        print('lstm_output_size', self.lstm_output_size)
        print('model_name', self.model_name)

if __name__ == '__main__':
    config = Config()
    print(config.max_features)
    print(config.maxlen)
    print(config.batch_size)