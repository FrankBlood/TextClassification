# -*- coding:utf8 -*-

from __future__ import print_function

class Config(object):

    def __init__(self, max_feature=200000, maxlen=2570,
                 batch_size=50, embedding_dims=50, nb_filter=250,
                 filter_length=3, hidden_dims=250, nb_epoch=2, dropout=0.2,
                 pool_length=4, lstm_output_size=70, model_name='model'):

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