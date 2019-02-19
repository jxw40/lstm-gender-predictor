
# coding: utf-8

from keras.models import Sequential
from keras import layers
from keras.models import model_from_json
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

import numpy as np
import sys
import os

#from keras.backend import manual_variable_initialization 
#manual_variable_initialization(True)

from keras import backend as K
K.clear_session()

#import keras
#keras.__version__
np.random.seed(42)

def preprocess():
    print("loading data...")
    with open("male.txt") as f:
        m_names = f.read().splitlines()
    
    with open("female.txt") as f:
        f_names = f.read().splitlines()
        
    bi_names = []
    for f_name in f_names:
	    if f_name in m_names:
		    bi_names.append(f_name)
#fs = set(f_names)
#ms = set(m_names)
#bi_names = fs & ms  # unisex names 
    
    m_names = [m_name.lower() for m_name in m_names if not m_name in bi_names]
    f_names = [f_name.lower() for f_name in f_names if not f_name in bi_names]
    
#m_names = ms - bi_names
    #print(len(m_names))
    
#f_names = fs - bi_names
    #print(len(f_names))
    
    #bi_names = [bi_name.lower() for bi_name in bi_names]
# m_names = [m_name.lower() for m_name in m_names]
#f_names = [f_name.lower() for f_name in f_names]

    # only male and only female names
    n_total = len(m_names) + len(f_names)
    
    max_len = max(len(max(m_names , key=len)), len(max(f_names , key=len)))
    
    chars = list(set("".join(m_names) + "".join(f_names)))
    chars.sort()
#chars = set("".join(m_names) + "".join(f_names))
#chars = ['e', '-', "'", 'c', 'y', 'd', 'l', 't', 'z', 'i', 'm', 'n', 'g', 'j', 'k', 'b', 'w', 'x', 'p', ' ', 's', 'q', 'f', 'u', 'a', 'o', 'r', 'h', 'v']
    n_chars = len(chars)
    
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    
    print('Corpus...')
    print("total entries: " , n_total)
    print("max_len: " , max_len)
    print('total chars:', n_chars)
    
    X_m = name2vec(m_names, len(m_names), max_len, n_chars, char_indices)
    X_f = name2vec(f_names, len(f_names), max_len, n_chars, char_indices)
    
    y_m = np.ones((len(m_names) , 1))
    y_f = np.zeros((len(f_names) , 1))
    
    X = np.concatenate((X_m, X_f), axis=0)
    y = np.concatenate((y_m, y_f), axis=0)

    return X, y, max_len, n_chars, char_indices, chars


def name2vec(names, n_instances, max_len, n_chars, char_indices):
    X = np.zeros((n_instances, max_len, n_chars))
    for i, name in enumerate(names):
        for t, char in enumerate(name):
            X[i, t, char_indices[char]] = 1.0
    return X


def train_test_split(X, y, n_train=6000):
    n_total = X.shape[0]
    idx = np.random.permutation(n_total)
    
    X_train = X[idx][:n_train]
    y_train = y[idx][:n_train]
    
    X_test = X[idx][n_train:]
    y_test = y[idx][n_train:]

    return X_train, y_train, X_test, y_test


def build(max_len, n_chars):

    print('Build model...')
    model = Sequential()
    model.add(layers.LSTM(128, return_sequences=True, dropout=0.2, input_shape=(max_len, n_chars)))
    #model.add(layers.Dropout(0.2))
    model.add(layers.LSTM(128, return_sequences=False, dropout=0.2))
    #model.add(layers.Dropout(0.2))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    return model


'''
history = model.fit(X_train, y_train,
                    epochs=n_epochs,
                    batch_size=128,
                    validation_split=0.2)
'''

'''
from matplotlib import pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
'''
def predict_gender(names, pred):
    print('Model predict...')
    p_male = 0
    p_female = 0
    for i in range(pred.shape[0]):
        if pred[i] > 0.5:
            p_male += 1
            print('%20s is Male'%names[i])
        else:
            p_female += 1
            print('%20s is Female'%names[i])
    print('predicted   male: ', p_male)
    print('predicted female: ', p_female)

if __name__ == "__main__":
    n_args = len(sys.argv)
    
    X, y, max_len, n_chars, char_indices, chars = preprocess()
#print(chars)
    X_train, y_train, X_test, y_test = train_test_split(X, y)

#filepath="weights.best.hdf5"
#filepath = 'weights.h5'
    filepath = 'gender_model_weights.h5'
    model = build(max_len, n_chars)
   
    if os.path.exists(filepath):
#model = model_from_json(open('model.json').read())
#model.load_weights(filepath)
        model = load_model(filepath)
        print('successfully retriving model...')
        
    if n_args == 2 and sys.argv[1] == 'train':
#        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#        callbacks_list = [checkpoint]

        model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
        n_epochs = 10
        history = model.fit(X_train, y_train,
                        epochs=n_epochs,
                        batch_size=64,
                        validation_split=0.2)
#        history = model.fit(X_train, y_train,
#                        epochs=n_epochs,
#                        batch_size=64,
#                        validation_split=0.2,
#                        callbacks=callbacks_list, verbose=0)
        print('after training')
        results = model.evaluate(X_test, y_test)
        print('test :',results)

#json_string = model.to_json()
#open('model.json', 'w').write(json_string)
#model.save_weights(filepath, overwrite=True)
        model.save(filepath)
        del model

    elif n_args == 3 and sys.argv[1] == 'predict':
        my_file_path = sys.argv[2]
        with open(my_file_path) as f:
            origin_names = f.read().splitlines()
            input_names = [input_name.lower() for input_name in origin_names]

        X_input = name2vec(input_names,  len(input_names), max_len, n_chars, char_indices)
    
        results = model.evaluate(X_test, y_test)
        print('checking test :',results)

        pred = model.predict(X_input)
        predict_gender(origin_names, pred)






