# -*- coding:utf8 -*-

from __future__ import print_function
import numpy as np
import codecs
import json
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem.porter import *
from string import punctuation

from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical

from config import Config
from data_loader import Data_Loader

class Pre_Process(object):

    def __init__(self):
        print('pre processing...')

    def process(self, config):
        """Process from raw file

        :param config:
        :param data_loader:
        :return: X_train, y_train, X_test, y_test, word_index

        """

        feature = []
        label = []

        print("Load data...")
        with codecs.open(config.data_path, encoding='utf8') as fp:
            for line in fp.readlines():
                json_data = json.loads(line.strip())
                text = json_data['text']
                stars = json_data['stars']
                text = self.text_to_wordlist(text.strip(), True)
                feature.append(text.strip())
                label.append([int(stars) - 1])

        print("all data is", len(feature))

        tokenizer = Tokenizer(num_words=config.max_features)
        tokenizer.fit_on_texts(feature)

        word_index = tokenizer.word_index
        print('Found %s unique tokens' % len(word_index))

        X_sequences = tokenizer.texts_to_sequences(feature)

        word_index = tokenizer.word_index
        print('Found %s unique tokens' % len(word_index))

        X = sequence.pad_sequences(X_sequences, maxlen=config.maxlen)
        print('X_train shape:', X.shape)
        y = to_categorical(label, num_classes=5)
        X_val = X[:830630]
        y_val = y[:830630]

        X_train = X[830630:-830630]
        y_train = y[830630:-830630]

        X_test = X[-830630:]
        y_test = X[-830630:]

        return np.array(X_train), np.array(y_train), np.array(X_val), np.array(y_val), np.array(X_test), np.array(y_test), word_index

    def get_tokenizer(self, config):

        feature = []
        label = []

        print("Load data...")
        with codecs.open(config.data_path, encoding='utf8') as fp:
            for line in fp.readlines():
                json_data = json.loads(line.strip())
                text = json_data['text']
                stars = json_data['stars']
                text = self.text_to_wordlist(text.strip(), True)
                feature.append(text.strip())
                label.append([int(stars) - 1])

        print("all data is", len(feature))

        tokenizer = Tokenizer(num_words=config.max_features)
        tokenizer.fit_on_texts(feature)

        word_index = tokenizer.word_index
        print('Found %s unique tokens' % len(word_index))

        return tokenizer, word_index

    def process_from_file(self, config, tokenizer, mode='train'):
        """Process from raw file

        :param config:
        :param data_loader:
        :return: X_train, y_train, X_test, y_test, word_index

        """

        if mode == 'train':
            path = config.train_path
        elif mode == 'validation':
            path = config.val_path
        elif mode == 'test':
            path = config.test_path
        else:
            print("What the fuck!")
            return None

        feature = []
        label = []

        print("Load data...")
        with codecs.open(path, encoding='utf8') as fp:
            for line in fp.readlines():
                json_data = json.loads(line.strip())
                text = json_data['text']
                stars = json_data['stars']
                text = self.text_to_wordlist(text.strip(), True)
                feature.append(text.strip())
                label.append([int(stars) - 1])

        print("all data is", len(feature))

        X_sequences = tokenizer.texts_to_sequences(feature)

        word_index = tokenizer.word_index
        print('Found %s unique tokens' % len(word_index))

        X = sequence.pad_sequences(X_sequences, maxlen=config.maxlen)
        print('X_train shape:', X.shape)
        y = to_categorical(label, num_classes=5)

        return np.array(X), np.array(y), word_index

    def process_from_file_old(self, config, data_loader):
        """Process from raw file

        :param config:
        :param data_loader:
        :return: X_train, y_train, X_test, y_test, word_index
        """

        X, y = data_loader.load_from_file(config)

        new_X = []
        for x in X:
            tmp = self.text_to_wordlist(x.strip(), True)
            new_X.append(tmp)
        X = new_X

        tokenizer = Tokenizer(num_words=config.max_features)
        tokenizer.fit_on_texts(X)

        X_train_sequences = tokenizer.texts_to_sequences(X)

        word_index = tokenizer.word_index
        print('Found %s unique tokens' % len(word_index))

        X = sequence.pad_sequences(X_train_sequences, maxlen=config.maxlen)
        print('X_train shape:', X.shape)
        y = to_categorical(y, num_classes=5)

        return np.array(X), np.array(y), word_index

    def text_to_wordlist(self, text, remove_stopwords=False, stem_words=False):
        # Clean the text, with the option to remove stopwords and to stem words.

        # Convert words to lower case and split them
        try:
            text = str(text.encode('utf8')).lower()
        except:
            text = str(text.encode('gbk')).lower()

        # Optionally, remove stop words
        if remove_stopwords:
            text = str(text).lower().split()
            stops = set(stopwords.words("english"))
            text = [w for w in text if not w in stops]
            text = " ".join(text)

        # Clean the text
        # text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
        # text = re.sub(r"[^A-Za-z0-9]", " ", text)
        text = re.sub(ur"\p{P}+", "", text)
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\(", " ", text)
        text = re.sub(r"\)", " ", text)
        text = re.sub(r"\（", " ", text)
        text = re.sub(r"\）", " ", text)
        text = re.sub(r"\:", " ", text)
        text = re.sub(r"\：", " ", text)
        text = re.sub(r"\;", " ", text)
        text = re.sub(r"\；", " ", text)
        text = re.sub(r"\！", " ", text)
        text = re.sub(r"\?", " ", text)
        text = re.sub(r"\？", " ", text)
        text = re.sub(r"\-", " ", text)
        text = re.sub(r"<br />", " ", text)
        text = re.sub(r"\'s", " is ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "cannot ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r"\,", " ", text)
        text = re.sub(r"\.", " ", text)
        text = re.sub(r"\!", " ! ", text)
        text = re.sub(r"\/", " ", text)
        text = re.sub(r"\\", " ", text)
        text = re.sub(r"\^", " ^ ", text)
        text = re.sub(r"\+", " + ", text)
        text = re.sub(r"\-", " - ", text)
        text = re.sub(r"\=", " = ", text)
        text = re.sub(r"\'", " ", text)
        text = re.sub(r"\‘", " ", text)
        text = re.sub(r"\’", " ", text)
        text = re.sub(r"\“", " ", text)
        text = re.sub(r"\”", " ", text)
        text = re.sub(r"\"", " ", text)
        text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
        text = re.sub(r":", " : ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r" u s ", " american ", text)
        text = re.sub(r"\0s", "0", text)
        text = re.sub(r" 9 11 ", "911", text)
        text = re.sub(r"e - mail", "email", text)
        text = re.sub(r"j k", "jk", text)
        text = re.sub(r"\s{2,}", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "cannot ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"I'm", "I am", text)
        text = re.sub(r" m ", " am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r"60k", " 60000 ", text)
        text = re.sub(r"quikly", "quickly", text)
        text = re.sub(r" usa ", " america ", text)
        text = re.sub(r" u s ", " america ", text)
        text = re.sub(r" uk ", " england ", text)
        text = re.sub(r"imrovement", "improvement", text)
        text = re.sub(r"intially", "initially", text)
        text = re.sub(r" dms ", "direct messages ", text)
        text = re.sub(r"demonitization", "demonetization", text)
        text = re.sub(r"actived", "active", text)
        text = re.sub(r"kms", " kilometers ", text)
        text = re.sub(r" cs ", " computer science ", text)
        text = re.sub(r" upvotes ", " up votes ", text)
        text = re.sub(r" iPhone ", " phone ", text)
        text = re.sub(r"\0rs ", " rs ", text)
        text = re.sub(r"calender", "calendar", text)
        text = re.sub(r"ios", "operating system", text)
        text = re.sub(r"programing", "programming", text)
        text = re.sub(r"bestfriend", "best friend", text)
        text = re.sub(r"iii", "3", text)
        text = re.sub(r"the us", "america", text)

        text = ' '.join([c for c in re.split('(\W+)?', str(text))
                         if (str(c).strip() not in punctuation) and (c.strip())])

        # Optionally, shorten words to their stems
        if stem_words:
            text = text.split()
            stemmer = SnowballStemmer('english')
            # stemmed_words = [stemmer.stem(word) for word in text]
            stemmed_words = []
            for word in text:
                try:
                    stemmed_words.append(stemmer.stem(word))
                except:
                    stemmed_words.append(word)
            text = " ".join(stemmed_words)

        # Return a list of words
        return text

def test():
    config = Config(max_feature=200000, maxlen=400, embedding_dims=300,
                    embedding_file='/home/hegx/Research/Quora_Question_Pairs/data/glove.840B.300d.txt')
    data_loader = Data_Loader()
    pre_process = Pre_Process()
    X, y, word_index = pre_process.process_from_file(config, data_loader)
    config.get_embedding_matrix(word_index)

if __name__ == "__main__":
    test()
