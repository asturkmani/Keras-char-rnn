from __future__ import print_function
import re
import urllib.request
import zipfile
import lxml.etree
import itertools
import numpy as np
import tensorflow as tf
import time
import pickle
import os
import random
import sys
import h5py
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras.utils import np_utils


# ## Import Data

# In[2]:

# Download Dataset
if not os.path.isfile('ted_en-20160408.zip'):
    urllib.request.urlretrieve("https://wit3.fbk.eu/get.php?path=XML_releases/xml/ted_en-20160408.zip&filename=ted_en-20160408.zip", filename="ted_en-20160408.zip")
    
    
# Extract documents   
with zipfile.ZipFile('ted_en-20160408.zip', 'r') as z:
    doc = lxml.etree.parse(z.open('ted_en-20160408.xml', 'r'))


# ## Character level LSTM language modelling

# In[3]:

corpus = ""
for document in doc.findall('//content'):
    corpus = corpus + "<s>" + document.text.lower() + "<e>"
print(len(corpus))
corpus = corpus[:4233275]

# In[4]:

chars_to_remove = ['+', ',', '-','/','<', '=', '>','@', '[', '\\', ']', '^', '_','\x80', '\x93', '\x94', '\xa0', '¡', '¢', '£', '²', 'º', '¿', 'à', 'á', 'â', 'ã', 'ä', 'å', 'æ', 'ç', 'è', 'é', 'ê', 'ë', 'ì', 'í', 'ï', 'ñ', 'ó', 'ô', 'ö', 'ø', 'ù', 'û', 'ü', 'ā', 'ă', 'ć', 'č', 'ē', 'ě', 'ī', 'ō', 'ť', 'ū', '˚', 'τ', 'ย', 'ร', 'อ', '่', '€', '∇', '♪', '♫', '你', '葱', '送', '–', '—', '‘', '’', '“', '”','0', '1', '2', '3', '4', '5', '6', '7', '8', '9','#', '$', '%', '&']
rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
corpus = re.sub(rx, '', corpus)


# In[5]:

chars = sorted(list(set(corpus)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
print(chars)


# In[6]:

# Split text into overlapping sentences with step size 3.
print('Splitting text into sequences...')
maxlen = 50
step = 1
sentences = []
next_chars = []
for i in range(0, len(corpus) - maxlen, step):
    sentences.append(corpus[i: i + maxlen])
    next_chars.append(corpus[i + maxlen])
print('number of sequences:', len(sentences))


# In[7]:

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# In[8]:

print(X.shape)
print(y.shape)


# In[11]:

# network parameters
N_HIDDEN = 256
N_HIDDEN2 = 256
N_HIDDEN3 = 256
LEARNING_RATE = 0.005
BATCH_SIZE = 64
EPOCHS = 40


# ### Build Model

# In[12]:

# build the model: a single LSTM
# print('Build model...')
# model = Sequential()
# model.add(LSTM(N_HIDDEN, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
# model.add(Dropout(0.5))
#model.add(LSTM(N_HIDDEN2, return_sequences=True,))
#model.add(Dropout(0.5))
#model.add(LSTM(N_HIDDEN3))
#model.add(Dropout(0.5))
#model.add(Dense(len(chars), activation ='softmax'))

optimizer = RMSprop(lr=LEARNING_RATE)

# load json and create model
json_file = open('model4.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("model4.h5")
print("Loaded model from disk")
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

#model.load_weights("model2.h5")

# In[13]:

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# In[14]:

# test and timing:
Xbatch = X
ybatch = y

print(Xbatch.shape)
print(ybatch.shape)
# In[16]:

    
    
for epoch in range(EPOCHS):
    
    t0 = time.time()
    model.fit(Xbatch, ybatch, batch_size=BATCH_SIZE, nb_epoch=1)
    t1 = time.time()
    total = t1-t0
    
    
    orig_stdout = sys.stdout
    f = open('out.txt', 'a+')
    sys.stdout = f
    
    print("------------- EPOCH" + str(epoch) + " ----------------")
    print('Time taken: ')
    print(total)
    
    
    # serialize model to JSON
    model_json = model.to_json()
    filename = "model" + str(epoch)
    with open(filename+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(filename+".h5")
    print("Saved model to disk")

    start_index = random.randint(0, len(corpus) - maxlen - 1)
    for diversity in [0.2, 1.0]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = corpus[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(800):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]
            generated += next_char
            sentence = sentence[1:] + next_char
        print(sentence)
    print()
    print()
    sys.stdout = orig_stdout
    f.close()
    
print(' 40 epochs completed ')


