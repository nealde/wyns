import collections
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gensim
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Convolution1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from keras.preprocessing.sequence import skipgrams

# load in our data
data = pd.read_csv("../core/data/tweet_global_warming.csv", encoding="latin")
print("Full dataset: {}".format(data.shape[0]))
data['existence'].fillna(value='Ambiguous', inplace = True) #replace NA's in existence with "ambiguous"
data['existence'].replace(('Y', 'N'), ('Yes', 'No'), inplace=True) #rename so encoder doesnt get confused
data = data.dropna() #now drop NA values
print("dataset without NaN: {}".format(data.shape[0]))

# split into x-y pairs after passing through word2vec:
X = data.iloc[:,0] #store tweets in X

labels = data.iloc[:,1]
confidence_interval = data.iloc[:,2]

# encode class as integers
encoder = LabelEncoder()
encoder.fit(labels)
encoded_Y = encoder.transform(labels)

# convert integers to one hot encoded
Y_one_hot = np_utils.to_categorical(encoded_Y)


# multiply one-hot by confidence intervals
Y=[]
for i, row in enumerate(confidence_interval):
    Y.append(row*Y_one_hot[i])
Y[0:5]
Y = np.array(Y)

# load in googles pretrained model
google = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin',binary=True)
vocab = google.vocab.keys()
total_vocab = len(vocab)
print ("Set includes", total_vocab, "words")
X_vecs = google.wv
del google


def read_data(data_file):
    for i, line in enumerate (data_file):
        # do some pre-processing and return a list of words for each review text
        yield gensim.utils.simple_preprocess (line)

tweet_vocab = list(read_data(data['tweet']))

vector_size = 300
window_size = 10
max_tweet_length=28

XX = np.zeros((len(X),max_tweet_length, vector_size))
for i in range(XX.shape[0]):
    for j, twit in enumerate(tweet_vocab[i]):
        if twit not in X_vecs:
            continue
        XX[i,j,:] = X_vecs[twit]
# print(XX[:-10,:,:])

# print(XX.shape)




inds = np.arange(XX.shape[0])
np.random.shuffle(inds)
# print(inds)
train = list(inds[:X.shape[0]*3//4])
test = list(inds[X.shape[0]*3//4:])
X_train = XX[train]
X_test = XX[test]

# reshape for 2d conv
X_train = X_train.reshape(*X_train.shape,1)
X_test = X_test.reshape(*X_test.shape,1)
Y_train = Y[train]
Y_test = Y[test]
print('final data processed')

# load up sentiment data from stanford
print('reading stanford data')
#cols = ['sentiment','id','date','query_string','user','text']
#data = pd.read_csv("../trainingandtestdata/training.1600000.processed.noemoticon.csv",header=None, names=cols, encoding = "ISO-8859-1")
#data['sentiment'] = (data['sentiment']-2)//2


# create function for shuffling the input data
#def shuffle(n):
#    inds = np.random.randint(0,len(data),n)
#    out = np.zeros((len(inds), 28, 300)).astype(np.float32)
#    for i, ind in enumerate(inds):
#        text = data['text'][ind].split(" ")
#        for j, word in enumerate(text):
#            if word not in X_vecs:
#                continue
#            if j > 27:
#                continue
#            out[i,j,:] = X_vecs[twit]
#    Y=[]
#    for i in np.array(data['sentiment'][inds]):
#        Y.append(Y_one_hot[i])
#    Y = np.array(Y)
#    return [out.astype(np.float32).reshape(*out.shape,1), Y.astype(np.int32)]
#
#xt, yt = shuffle(10000) # set xtest and ytest for
#print(yt.shape)
#
## make the model
#from keras.layers import Dropout, Convolution2D, MaxPooling2D
#
#top_words = 1000
#max_words = 150
#filters = 32 #filter = 1 x KERNEL
#batch_size = 128
#
#inpurt_shape = (*X_train.shape[1:],1)
#print(inpurt_shape)
## create the model
#model = Sequential()
#
#model.add(Convolution2D(32, kernel_size=9, activation='elu', padding='same',
#                 input_shape=inpurt_shape))
#model.add(MaxPooling2D(pool_size=5))
#model.add(Convolution2D(filters=64, kernel_size=9, padding='same', activation='elu'))
#model.add(MaxPooling2D(pool_size=2))
#model.add(Convolution2D(filters=64, kernel_size=9, padding='same', activation='elu'))
#model.add(MaxPooling2D(pool_size=2))
#model.add(Convolution2D(filters=filters, kernel_size=9, padding='same', activation='elu'))
#model.add(Flatten())
#model.add(Dense(250, activation='elu'))
#model.add(Dropout(0.5))
#model.add(Dense(250, activation='elu'))
#model.add(Dropout(0.5))
#model.add(Dense(3, activation='linear')) #change from logistic
#model.compile(loss='mse', optimizer='adam', metrics=['accuracy','mse'])
#print(model.summary())
#
#def myGenerator():
#    while 1:
#        x, y = shuffle(50000)
#        for i in range(int(50000/batch_size)):
#            yield [x[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size,:]]
#
## Fit the model
#
#model.fit_generator(myGenerator(),
#                    steps_per_epoch = 50000/batch_size,
#                    epochs = 200,
#                    verbose=1)
##                   validation_data=([xt, xyt], yt))
#
save = 1
#if save == 1:
#    model.save('new_generator.h5')

model = load_model('new_generator.h5')
# now, train on real data:
model.fit(X_train,
          Y_train,
          validation_data=(X_test, Y_test),
          epochs=20,
          batch_size=128,
          verbose=1)
if save == 1:
    model.save('final.h5')
