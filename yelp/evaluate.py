# -*- coding:utf8 -*-

from __future__ import print_function
from pre_process import Pre_Process
from model import *
import argparse
import os
import sys

from config import Config

FLAGS = None

def evaluate(config, bst_model_path):
    pre_process = Pre_Process()
    _, _, _, _, X_test, y_test, word_index = pre_process.process(config)
    embedding_matrix = config.get_embedding_matrix(word_index)

    print('Evaluating...')

    if config.model_name == 'bidirectional_lstm':
        model = bidirectional_lstm(config, embedding_matrix)

    elif config.model_name == 'cnn':
        model = cnn(config, embedding_matrix)

    elif config.model_name == 'cnn_lstm':
        model = cnn_lstm(config, embedding_matrix)

    elif config.model_name == 'lstm':
        model = lstm(config, embedding_matrix)

    elif config.model_name == 'cnn_based_rnn':
        model = cnn_based_rnn(config, embedding_matrix)

    elif config.model_name == 'complex_cnn_based_rnn':
        model = complex_cnn_based_rnn(config, embedding_matrix)

    elif config.model_name == 'cnn_add_rnn':
        model = cnn_add_rnn(config, embedding_matrix)

    elif config.model_name == 'cnn_inner_rnn':
        model = cnn_inner_rnn(config, embedding_matrix)

    elif config.model_name == 'rnn_inner_rnn':
        model = rnn_inner_rnn(config, embedding_matrix)

    elif config.model_name == 'rnn_cat_rnn':
        model = rnn_cat_rnn(config, embedding_matrix)

    elif config.model_name == 'pre_attention_inner_rnn':
        model = pre_attention_inner_rnn(config, embedding_matrix)

    else:
        print("What the FUCK!")
        return

    print('bst_model_path:', bst_model_path)

    if os.path.exists(bst_model_path):
        model.load_weights(bst_model_path)

    print('test:', model.evaluate(X_test, y_test, batch_size=config.batch_size))

def main():
    print('Configurations:')
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_features', type=int, help='max features.')
    parser.add_argument('--maxlen', type=int, help='max len.')
    parser.add_argument('--embedding_dims', type=int, help='embedding dims.')
    parser.add_argument('--dropout', type=float, help='dropout.')
    parser.add_argument('--model_name', type=str, help='model name.')

    parser.add_argument('--batch_size', type=int, default=10, help='batch size.')
    parser.add_argument('--nb_filter', type=int, default=None, help='nb filter.')
    parser.add_argument('--filter_length', type=int, default=None, help='filter length.')
    parser.add_argument('--hidden_dims', type=int, default=None, help='hidden_dims.')
    parser.add_argument('--nb_epoch', type=int, default=None, help='nb epoch.')
    parser.add_argument('--pool_length', type=int, default=None, help='pool length.')
    parser.add_argument('--lstm_output_size', type=int, default=None, help='lstm output size.')
    parser.add_argument('--embedding_file', type=str, default=None, help='embeddin file.')
    parser.add_argument('--word_vocb_path', type=str, default=None, help='word vocb path.')
    parser.add_argument('--train_path', type=str, default=None, help='train path.')
    parser.add_argument('--test_path', type=str, default=None, help='train path.')
    parser.add_argument('--model_path', type=str, default=None, help='the model path.')
    sys.stdout.flush()
    FLAGS, unparsed = parser.parse_known_args()
    print('FLAGS', FLAGS)
    print(FLAGS.max_features)
    config = Config(max_feature=FLAGS.max_features, maxlen=FLAGS.maxlen,
                    batch_size=FLAGS.batch_size, embedding_dims=FLAGS.embedding_dims,
                    nb_filter=FLAGS.nb_filter, filter_length=FLAGS.filter_length,
                    hidden_dims=FLAGS.hidden_dims, nb_epoch=FLAGS.nb_epoch,
                    dropout=FLAGS.dropout, pool_length=FLAGS.pool_length,
                    lstm_output_size=FLAGS.lstm_output_size, model_name=FLAGS.model_name,
                    embedding_file=FLAGS.embedding_file, word_vocb_path=FLAGS.word_vocb_path)
    config.print_config()
    evaluate(config, FLAGS.model_path)

if __name__ == '__main__':
    main()