from __future__ import absolute_import, division, print_function
import wyns as core
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def test_baseline_model():
    X, Y = core.data_setup()
    model = core.baseline_model()
    X_train, X_test, y_train, y_test = train_test_split(X, Y)

    # Fit the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2,
              batch_size=128, verbose=0)

    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    accuracy = scores[1] * 100
    print("Accuracy: %.2f%%" % accuracy)
    if accuracy < 60:
        raise Exception('Baseline model accuracy is too low')
    else:
        pass


def test_load_data():
    data_file_name = "tweet_global_warming.doc"
    data = core.load_data("tweet_global_warming.csv")
    try:
        data = core.load_data(data_file_name)
    except(AssertionError):
        print("Upload a csv file")
    else:
        raise Exception("Code only handles csv or hdf5 files")
    if type(data) == pd.DataFrame:
        pass
    else:
        raise Exception("Output format should be Pandas DataFrame")


def test_data_setup():
    top_words = "100"
    max_words = "100"
    max_words2 = 100
    X, Y = core.data_setup(100, max_words2)
    try:
        core.data_setup(top_words, max_words)
    except(AssertionError):
        print("Inputs must be type int")
    if X.shape[1] == max_words2:
        pass
    else:
        raise Exception('The column length of X should match with maximum \
                        words')
    if list(np.unique(Y)) == [0, 1]:
        pass
    else:
        raise Exception('one hot encoded should not contain other values \
                        except 0 and 1')


def test_read_data():
    file = pd.read_csv('../data/tweet_global_warming.csv', encoding='latin')
    data = core.read_data(file['tweet'])
    words = list(data)
    try:
        core.read_data(file)
    except(AssertionError):
        print("Input must be a pandas Series")
    for i in file['tweet']:
        if isinstance(i, str):
            pass
        else:
            raise Exception('Input should be a string of sentences')
    if len(words) == len(file):
        pass
    else:
        raise Exception('The processed output length should be equal \
                        to the number of tweets')


def test_build_dataset():
    vocab = pd.DataFrame({'col1': ['hey', 'there', 'human']})
    try:
        core.build_dataset(vocab, 1)
    except(AssertionError):
        print("Inputs must be list. Use read_data if using a pd.DataFrame)")
    else:
        raise Exception("Make sure n_words is type int")
