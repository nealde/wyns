from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM, GRU
from keras.layers.recurrent import SimpleRNN
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from keras.optimizers import RMSprop
import numpy as np
from tqdm import tqdm
import string
import pandas as pd
from keras.utils import np_utils
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
# dfile = 'shakespear.txt'

# with open(dfile,'r') as f:
#     raw = f.read()

# lowers = string.ascii_lowercase

# k = set(raw.lower()) - set(lowers)
# ''.join(sorted(k))

# extra = "\n !?';,."
# allowed = set(lowers + extra)
# print(allowed)
# from collections import Counter, defaultdict

# D = dict([(k,k) if k in allowed else (k, ' ') for k in set(raw.lower())])
# keys = ''.join(D.keys())
# vals = ''.join([D[k] for k in keys])
# DD = str.maketrans(keys,vals)

# data = raw.lower().translate(DD)
# print(data)

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

# print(X)
X = X.str.lower()
# get all unique characters:

unique = []
for i in X:
    for j in set(i):
        if j not in unique:
            unique.append(j)
print(unique)


chars = list(unique)

# change unroll length
maxlen=20
step = 1
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
print(char_indices)


# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(Embedding(len(chars), 48, input_length=maxlen))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(64))
model.add(Dense(3))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

epochs = 10
num_blocks = 100
# print(list(X.iloc[0]))

data = []
for i in X:
    data.append(list(i))
# print(len(data))

# print(data.shape)

x_list = []
y_list = []
for i in range(len(data)):
    for j in range(0, len(data[i])-maxlen, step):
        twit = data[i][j:j + maxlen]
        # turn the twit into numbers
        num_twit = []
        for char in twit:
            num_twit.append(char_indices[char])
        x_list.append(num_twit)
        y_list.append(Y_one_hot[i])
x_list = np.array(x_list)
y_list = np.array(y_list)

import random
train = sorted(random.sample(range(x_list.shape[0]), int(x_list.shape[0]*(3.0/4.))))
test = []
count = 0
for i in range(x_list.shape[0]):
    if train[count] == i:
        count += 1
    else:
        test.append(i)

# print(test)

# test = [t for t in range(x_list.shape[0]) if t not in train]

x1 = x_list[train,:]
xt = x_list[test,:]
y1 = y_list[train,:]
yt = y_list[test,:]

# print(x_list.shape)
# print(y_list.shape)
# print(x_list[0])

model.fit(x1, y1,
          batch_size = 5120,
         epochs = 200,
         verbose = 0) 
model.fit(x_list, y_list,
          batch_size = 5120,
         epochs = 1,
         verbose = 1,
         validation_data = [xt, yt]) 

model.save('neal_rnn_85.h5')

def classify_twit(twit, model, maxlen=20, step=1):
    assert isinstance(twit, str)
    twit = list(twit)
    twit_list = []
    for i in range(0, len(twit)-maxlen, step):
        tweet = twit[i:i+maxlen]
        num_twit = []
        for char in tweet:
            num_twit.append(char_indices[char])
        twit_list.append(num_twit)
    twit_list = np.array(twit_list)
    pred = model.predict(twit_list)
    print(pred.shape)
    return np.array([pred[:,i].mean() for i in range(pred.shape[1])])

print(X.iloc[0])
classify_twit(X.iloc[0], model)

# model.fit_generator(myGenerator(), 
#                     steps_per_epoch = len(x_list), 
#                     epochs = 10, 
#                     verbose=1)

# model.evaluate(x_list[0], y_list[0])



class bard(object):
    def __init__(self,model,primer = 'the quick brown fox jumps over the lazy ', maxlen = 20, numchar = 34, chars = chars, diversity = 0.5):
        self.model = model
        self.text = primer[-maxlen:].lower()
        assert set(self.text).issubset(set(chars))
        self.diversity = diversity
        self.chars = chars
        self.onehot = np.zeros([1,maxlen,numchar],dtype=np.uint8)
        for i,p in enumerate(primer[::-1]):
            self.onehot[0,maxlen-i-1,self.chars.index(p)] = 1
        self.dense = np.argmax(self.onehot,axis=2)
    def sample(self, probs, diversity=0.5):
        probs = np.asarray(probs).astype('float64')
        exp_preds = np.exp(np.log(probs)/diversity)
        preds = exp_preds / sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)
    def step(self,n=1,verbose=True):
        for i in range(n):
            probs = self.model.predict(self.dense)[0]
            idx = self.sample(probs, self.diversity)
            self.text += self.chars[idx]
            self.onehot[0,:-1] = self.onehot[0,1:]
            self.onehot[0,-1] = 0
            self.onehot[0,-1,self.chars.index(self.text[-1])] = 1
            self.dense = np.argmax(self.onehot,axis=2)
        if verbose:
            print(self.text)

bill = bard(model)

bill.step(40,verbose=True)

b2 = bard(model,diversity = .5, primer = ''.join(data[1,1000:1040]))
b2.step(10)
b2.step(1000)
