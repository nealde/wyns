import numpy as np
import pandas as pd
import gensim
import re

from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

filename = 'tweet_global_warming.csv'
df = pd.read_csv(filename, encoding='latin')
df.head()
model_path = "../../examples/GoogleNews-vectors-negative300.bin"
word_vector_model = \
    gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)


def normalize(txt, vocab=None, replace_char=' ',
              max_length=300, pad_out=False,
              to_lower=True, reverse=False,
              truncate_left=False, encoding=None,
              letters_only=False):

    txt = txt.split()
    # Remove HTML
    # This will keep characters and other symbols
    txt = [re.sub(r'http:.*', '', r) for r in txt]
    txt = [re.sub(r'https:.*', '', r) for r in txt]

    txt = (" ".join(txt))
    # Remove non-emoticon punctuation and numbers
    txt = re.sub("[.,!0-9]", " ", txt)
    if letters_only:
        txt = re.sub("[^a-zA-Z]", " ", txt)
    txt = " ".join(txt.split())
    # store length for multiple comparisons
    txt_len = len(txt)

    if truncate_left:
        txt = txt[-max_length:]
    else:
        txt = txt[:max_length]
    # change case
    if to_lower:
        txt = txt.lower()
    # Reverse order
    if reverse:
        txt = txt[::-1]
    # replace chars
    if vocab is not None:
        txt = ''.join([c if c in vocab else replace_char for c in txt])
    # re-encode text
    if encoding is not None:
        txt = txt.encode(encoding, errors="ignore")
    # pad out if needed
    if pad_out and max_length > txt_len:
        txt = txt + replace_char * (max_length - txt_len)
    if txt.find('@') > -1:
        for i in range(len(txt.split('@')) - 1):
            try:
                if str(txt.split('@')[1]).find(' ') > -1:
                    to_remove = '@' + str(txt.split('@')[1].split(' ')[0]) +\
                                " "
                else:
                    to_remove = '@' + str(txt.split('@')[1])
                txt = txt.replace(to_remove, '')
            except:
                pass
    return txt


def tweet_to_sentiment(tweet):
    # Review is coming in as Y/N/NaN
    # This then cleans the summary and review and gives it a
    # positive or negative value
    norm_text = normalize(tweet[0])
    if tweet[1] in ('Yes', 'Y'):
        return ['positive', norm_text]
    elif tweet[1] in ('No', 'N'):
        return ['negative', norm_text]
    else:
        return ['other', norm_text]


def clean_tweet(tweet):
    norm_text = normalize(tweet[0])
    return [tweet[1], tweet[2], norm_text, tweet[3], tweet[4],
            tweet[5], tweet[0]]


df = pd.read_csv(filename, encoding='latin')
data = []
for index, row in df.iterrows():
    data.append(tweet_to_sentiment(row))

twitter = pd.DataFrame(data, columns=['Sentiment', 'clean_text'], dtype=str)

# Now go from the pandas into lists of text and labels
text = twitter['clean_text'].values
# mapping of the labels with dummies (has headers)
labels_0 = pd.get_dummies(twitter['Sentiment'])
# print(labels_0[:10], twitter['Sentiment'].iloc[:10])
# labels = labels_0.values
labels = labels_0.values[:, [0, 2]]  # removes the headers
print(labels)
# Perform the Train/test split
X_train_, X_test_, Y_train_, Y_test_ = train_test_split(text,
                                                        labels,
                                                        test_size=0.2,
                                                        random_state=42)

# Now for a simple bidirectional LSTM algorithm
# we set our feature sizes and train a tokenizer
# First we Tokenize and get the data into a form that
# the model can read - this is BoW
# In this cell we are also going to define some
# of our hyperparameters
max_features = 2000
max_len = 2000
words_len = 30
batch_size = 32
embed_dim = 300
lstm_out = 140

dense_out = len(labels[0])  # length of features
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(X_train_)
X_train = tokenizer.texts_to_sequences(X_train_)
X_train = pad_sequences(X_train, maxlen=words_len, padding='post')
print(X_train[:, -1].mean())
X_test = tokenizer.texts_to_sequences(X_test_)
X_test = pad_sequences(X_test, maxlen=words_len, padding='post')
word_index = tokenizer.word_index
# print(len(word_index))


df = pd.read_csv("tweets.txt", delimiter="~~n~~", engine="python")
data = []
for index, row in df.iterrows():
    data.append(clean_tweet(row))
twitter = pd.DataFrame(data, columns=['long', 'lat', 'clean_text',
                                      'time', 'retweets', 'location',
                                      'raw_tweet'], dtype=str)
to_predict_ = twitter['clean_text'].values

# Now for a simple bidirectional LSTM algorithm we set our
# feature sizes and train a tokenizer
# First we Tokenize and get the data into a form that the
# model can read - this is BoW
# In this cell we are also going to define some of our
# hyperparameters

max_features = 2000
max_len = 30
batch_size = 32
embed_dim = 300
lstm_out = 140

dense_out = len(labels[0])  # length of features
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(to_predict_)
to_predict = tokenizer.texts_to_sequences(to_predict_)
to_predict = pad_sequences(to_predict, maxlen=max_len, padding='post')
word_index = tokenizer.word_index

model = load_model("climate_sentiment_m6.h5")

predictions = model.predict(to_predict)

print("negative predictions: {}".format(sum(np.round(predictions)[:, 0])))
print("positive predictions: {}".format(sum(np.round(predictions)[:, 1])))

df_out = pd.DataFrame([twitter['long'], twitter['lat'], twitter['clean_text'],
                      twitter['time'], twitter['retweets'],
                      twitter['location'], twitter['raw_tweet'],
                      predictions[:, 0], predictions[:, 1]]).T
df_out = df_out.rename(index=str, columns={"Unnamed 0": "negative",
                                           "Unnamed 1": "positive"})
print(df_out.shape)
df_out.head()

df_out.to_csv("sample_prediction.csv", index=False)
