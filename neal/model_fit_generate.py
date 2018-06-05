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

dfile = 'shakespear.txt'

with open(dfile,'r') as f:
    raw = f.read()

lowers = string.ascii_lowercase

k = set(raw.lower()) - set(lowers)
''.join(sorted(k))

extra = "\n !?';,."
allowed = set(lowers + extra)
print(allowed)
from collections import Counter, defaultdict

D = dict([(k,k) if k in allowed else (k, ' ') for k in set(raw.lower())])
keys = ''.join(D.keys())
vals = ''.join([D[k] for k in keys])
DD = str.maketrans(keys,vals)

data = raw.lower().translate(DD)
print(data)

# collect repeated spaces and newlines
while '  ' in data:
    data = data.replace('  ',' ')
while '\n\n' in data:
    data = data.replace('\n\n','\n')
while '\n \n' in data:
    data = data.replace('\n \n','\n')

chars = list(lowers+extra)

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
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])

epochs = 10
num_blocks = 100


# truncate the data and reshape
data = data[:-(len(data)%num_blocks)]
data = np.array(list(data)).reshape([num_blocks,-1])
print(data)
print(len(data[0]))

# print(data.shape)


x_list = []
y_list = []
for b in tqdm(range(num_blocks)):
    sentences = []
    next_chars = []
    for i in range(0, len(data[b]) - maxlen, step):
        sentences.append(data[b,i: i + maxlen])
        next_chars.append(data[b,i + maxlen])
    # stick with dense encoding
    X = np.zeros([len(sentences),maxlen],dtype=np.uint8)
    # encode all in one-hot
    Y = np.zeros([len(sentences),len(chars)],dtype=np.uint8)
    i = 0
    for t, char in enumerate(sentences[0]):
        X[i, t]= char_indices[char]
        Y[i, char_indices[next_chars[i]]] = 1
    for i, sentence in enumerate(sentences[1:]):
            X[i+1, :-1] = X[i, 1:]
            X[i+1, -1] = char_indices[next_chars[i]]
            Y[i+1, char_indices[next_chars[i+1]]] = 1
    x_list.append(X)
    y_list.append(Y)
print(len(sentences))

# print(x_list[0])
for i in range(num_blocks):
    print(x_list[i].shape)
# X = np.array(x_list)
# Y = np.array(y_list)
# print(X.shape)
# print(X[0].shape)

def myGenerator():
    while 1:
        for i in range(len(x_list)):
            yield [x_list[i], y_list[i]]

model.fit_generator(myGenerator(), 
                    steps_per_epoch = len(x_list), 
                    epochs = 100, 
                    verbose=1)
model.fit_generator(myGenerator(), 
                    steps_per_epoch = len(x_list), 
                    epochs = 1, 
                    verbose=1)
# model.evaluate(x_list[0], y_list[0])

# model.save_weights('bardicweights.h5')
# model.load_weights('bardicweights.h5')
model.save('shakespeare_model.h5')



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
