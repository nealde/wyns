from __future__ import absolute_import, division, print_function
import wyns as core
import pandas as pd
import unittest
from sklearn.model_selection import train_test_split


class testKerasModels(unittest.TestCase):

    def test_baseline_model(self):
        X, Y = core.data_setup()
        model = core.baseline_model()
        X_train, X_test, y_train, y_test = train_test_split(X, Y)

        # Fit the model
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2,
                  batch_size=128, verbose=0)

        # Final evaluation of the model
        scores = model.evaluate(X_test, y_test, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))

    def test_benchmark(self):
        core.Benchmark.run(self.test_baseline_model)


def test_load_data():
    data_file_name = "tweet_global_warming.doc"
    try:
        core.load_data(data_file_name)
    except(AssertionError):
        print("Upload a csv file")
    else:
        raise Exception("Code only handles csv or hdf5 files")


def test_data_setup():
    top_words = "100"
    max_words = "100"
    try:
        core.data_setup(top_words, max_words)
    except(AssertionError):
        print("Inputs must be type int")


def test_read_data():
    data_file = [1, 2, 3]
    try:
        core.read_data(data_file)
    except(AssertionError):
        print("Input must be a pandas DataFrame")


def test_build_dataset():
    vocab = pd.DataFrame({'col1': ['hey', 'there', 'human']})
    try:
        core.build_dataset(vocab, 1)
    except(AssertionError):
        print("Inputs must be list. Use read_data if using a pd.DataFrame)")
    else:
        raise Exception("Make sure n_words is type int")


if __name__ == '__main__':
    unittest.main()
