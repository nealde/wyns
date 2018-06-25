import collections
import gensim
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from keras.models import load_model
from os.path import dirname, join
import sys
import time
import statistics


def load_data(data_file_name, h5File=False):
    """
    Loads data from module_path/data/data_file_name.

    Parameters
    ----------
    data_file_name : string
        name of csv file to be loaded from module_path/data/
        data_file_name.
    h5File : boolean, optional, default = False
        if True opens hdf5 file

    Returns
    -------
    data : Pandas DataFrame
        Loaded data in a dataframe
    """
    module_path = dirname(__file__)
    if h5File:
        data = load_model(join(module_path, 'data', data_file_name))
    else:
        assert data_file_name[-3:] == "csv"
        with open(join(module_path, 'data', data_file_name), 'rb') as csv_file:
            data = pd.read_csv(csv_file, encoding='latin1')
    return data


def data_setup(top_words=1000, max_words=150):
    """
    Preprocesses the twitter climate data. Does things like changes output
    to one hot encoding, performs word embedding/padding

    Parameters
    ----------
    top_words : int
        The maximum number of words to keep, based on word frequency
        in the training corpus
    max_words : int
        The maximum length of all pad sequences

    Returns
    -------
    X : numpy array
        The embedded word vectors for all tweets
    Y : numpy array
        The one hot encoded labels
    """
    assert type(top_words) == int
    assert type(max_words) == int
    data = load_data("tweet_global_warming.csv")
    print("Full dataset: {}".format(data.shape[0]))
    # replace NA's in existence with "ambiguous"
    data['existence'].fillna(value='ambiguous',
                             inplace=True)
    # rename so encoder doesnt get confused
    data['existence'].replace(('Y', 'N'), ('Yes', 'No'),
                              inplace=True)
    data = data.dropna()  # now drop NA values
    print("dataset without NaN: {}".format(data.shape[0]))
    X = data.iloc[:, 0]
    Y = data.iloc[:, 1]
    print("Number of unique words: {}".format(len(np.unique(np.hstack(X)))))

    # one hot encoding = dummy vars from categorical var
    # Create a one-hot encoded binary matrix
    # N, Y, Ambig
    # 1, 0, 0
    # 0, 1, 0
    # 0, 0, 1

    # encode class as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)

    # convert integers to one hot encoded
    Y = np_utils.to_categorical(encoded_Y)

    # convert X to ints (y is already done)
    token = Tokenizer(num_words=top_words,
                      filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True,
                      split=' ', char_level=False, oov_token=None)
    token.fit_on_texts(texts=X)
    X = token.texts_to_sequences(texts=X)
    X = sequence.pad_sequences(X, maxlen=max_words)
    return X, Y


def baseline_model(top_words=1000, max_words=150, filters=32):
    """
    Preprocesses the twitter climate data. Does things like changes output
    to one hot encoding, performs word embedding/padding

    Parameters
    ----------
    top_words : int
        The maximum number of words to keep, based on  word frequency
        in the training corpus
    max_words : int
        The maximum length of all pad sequences
    filters : int
        The number of output filters in the convolution

    -------
    model : object
        The Keras model object used to fit training data
    """
    model = Sequential()
    model.add(Embedding(top_words + 1, filters,
                        input_length=max_words))
    model.add(Convolution1D(filters=filters, kernel_size=3, padding='same',
                            activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(250, activation='relu'))
    model.add(Dense(3, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    return model


def read_data(data_file):
    """
    Takes a data file and returns a vector containing a list of words using
    gensim preprocessing

    Parameters
    ----------
    data_file : pandas.core.series.Series
        Tweets in a Pandas DataFrame column

    Returns
    -------
\
    """

    assert type(data_file) == pd.DataFrame
    for i, line in enumerate(data_file):
        yield gensim.utils.simple_preprocess(line)


def build_dataset(vocab, n_words):
    """
    Process the top n_words of the vocab and outputs a token and count for
    each word as well as dictionaries for forward and reverse lookup

    Parameters
    ----------
    vocab : list
        A list of words after the tweet preprocessing
    n_words : int
        The maximum number of words to keep, based on word frequency
        in the training corpus

    Returns
    -------
    token : list
        A list of integers of tokenized words based on word frequency
    count : list
        The appearance count of each word starting from the most
        frequent word
    dictionary : dict
        A index dictionary of words starting from the most frequent
        word. The dict keys are words and values are words
    reversed_dictionary : dict
        The reversed dictionary with keys being the indexes and
        words being the values
    """
    assert type(vocab) == list
    assert type(n_words) == int
    count = [['UNK', -1]]  # UNK = unknown --> words filtered out by n_words
    count.extend(collections.Counter(vocab).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    token = list()
    unk_count = 0
    for word in vocab:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK'] assigned to 0
            unk_count += 1
        token.append(index)  # outputs a list of integers that represent words
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return token, count, dictionary, reversed_dictionary


class Benchmark:
    """
    benchmark method used by the unittests
    """
    @staticmethod
    def run(function):
        timings = []
        stdout = sys.stdout
        for i in range(5):
            sys.stdout = None
            startTime = time.time()
            function()
            seconds = time.time() - startTime
            sys.stdout = stdout
            timings.append(seconds)
            mean = statistics.mean(timings)
            print("{} {:3.2f} {:3.2f}".format(
                1 + i, mean,
                statistics.stdev(timings, mean) if i > 1 else 0))
