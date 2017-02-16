import re
import urllib.request
import zipfile
import lxml.etree
import itertools
import numpy as np
import tensorflow as tf
import pickle
import os
import random
import sys
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras.models import load_model


# ## Import Data

# In[ ]:

# Download Dataset
if not os.path.isfile('ted_en-20160408.zip'):
    urllib.request.urlretrieve("https://wit3.fbk.eu/get.php?path=XML_releases/xml/ted_en-20160408.zip&filename=ted_en-20160408.zip", filename="ted_en-20160408.zip")
    
    
# Extract documents   
with zipfile.ZipFile('ted_en-20160408.zip', 'r') as z:
    doc = lxml.etree.parse(z.open('ted_en-20160408.xml', 'r'))


# ## Character level LSTM language modelling

# In[ ]:

corpus = ""
for document in doc.findall('//content'):
    corpus = corpus + "<s>" + document.text.lower() + "<e>"
print(len(corpus))


# In[ ]:

chars = sorted(list(set(corpus)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))


# In[ ]:

# Split text into overlapping sentences with step size 3.
print('Splitting text into sequences...')
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(corpus) - maxlen, step):
    sentences.append(corpus[i: i + maxlen])
    next_chars.append(corpus[i + maxlen])
print('number of sequences:', len(sentences))


# In[ ]:

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# In[ ]:

print(X.shape)
print(y.shape)


# In[ ]:

#take subset of data to check if model works as expected
Xtemp = X
ytemp = y

print(Xtemp.shape)
print(ytemp.shape)


# In[ ]:

# network parameters
N_HIDDEN = 128
LEARNING_RATE = 0.01
BATCH_SIZE = 256
EPOCHS = 1


# ### Build Model

# In[ ]:

# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(N_HIDDEN, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=LEARNING_RATE)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


# In[ ]:

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# In[ ]:

# train the model, output generated text after each iteration
for iteration in range(1, 30):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    model.fit(X, y, batch_size=BATCH_SIZE, nb_epoch=EPOCHS)

    start_index = random.randint(0, len(corpus) - maxlen - 1)

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = corpus[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()


# In[ ]:

# save trained model
model.save('my_model.h5')


# In[ ]:



