

def train_model(filename, save=1):
    '''The main function that creates and trains the model(s)'''
    import numpy as np
    import pandas as pd
    import h5py 
    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, AlphaDropout
    from keras.preprocessing.text import Tokenizer
    import tensorflow as tf
    print('keras', keras.__version__)
    print('tensorflow', tf.__version__)
    
    # read in the data:
    print('reading data...')
    x_sets = []
    y_sets = []
    xt_sets = []
    yt_sets = []
    with h5py.File('twitter_data.hdf5', 'r') as f:
        for i in range(5):
            x_sets.append(f['x%i' % i][:])
            xt_sets.append(f['xt%i' % i][:])
            y_sets.append(f['y%i' % i][:])
            yt_sets.append(f['yt%i' % i][:])
        xtest = f['xtest'][:]
        ytest = f['ytest'][:]
    print(len(x_sets))
    for i in range(len(x_sets)):
        x_sets[i] = [x.decode('utf-8') for x in x_sets[i]]
        xt_sets[i] = [x.decode('utf-8') for x in xt_sets[i]]
    xtest = [x.decode('utf-8') for x in xtest]
    
    all_text = flat_list = [item for sublist in x_sets for item in sublist]
    tokenizer = Tokenizer(nb_words = 500)
    tokenizer.fit_on_texts(all_text)
    vocab_size = 500
    
    # create the model and train
    model = Sequential()
    model.add(Dense(10, input_dim=vocab_size))
    model.add(Dense(10, activation='selu'))
    model.add(Dense(10, activation='selu'))
    model.add(AlphaDropout(0.2))
    model.add(Dense(3, activation='sigmoid'))

    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['acc'])
    i = 0
    model.fit(tokenizer.texts_to_matrix(x_sets[i]), y=y_sets[i], 
              batch_size=200, 
              nb_epoch=100, 
              verbose=1, 
              class_weight = [1/.5,1/.2,1/.3],
             validation_data = [tokenizer.texts_to_matrix(xt_sets[i]), yt_sets[i]])
    
    if save == 1:
        model.save(filename+'.h5')
    return model
train_model('model')