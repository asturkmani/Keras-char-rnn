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
from keras.layers import Dense, Activation, Dropout, LSTM, GRU
from keras.optimizers import RMSprop, Adam
from keras.utils.data_utils import get_file
from keras.utils import np_utils
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import LearningRateScheduler


# ## Import Data

# In[129]:

FLAGS = tf.flags
FLAGS.data_path = "talks.txt"
FLAGS.maxlen = 50
FLAGS.batch_size = 32


# In[100]:

# Download Dataset
if not os.path.isfile('ted_en-20160408.zip'):
    urllib.request.urlretrieve("https://wit3.fbk.eu/get.php?path=XML_releases/xml/ted_en-20160408.zip&filename=ted_en-20160408.zip", filename="ted_en-20160408.zip")


# Extract documents   
with zipfile.ZipFile('ted_en-20160408.zip', 'r') as z:
    doc = lxml.etree.parse(z.open('ted_en-20160408.xml', 'r'))


# In[102]:


# corpus = ""
# totalcorpus = ""
# i=0
# chars_to_remove = ['+', ',', '-','/','<', '=', '>','@', '[', '\\', ']', '^', '_','\x80', '\x93', '\x94', '\xa0', '¡', '¢', '£', '²', 'º', '¿', 'à', 'á', 'â', 'ã', 'ä', 'å', 'æ', 'ç', 'è', 'é', 'ê', 'ë', 'ì', 'í', 'ï', 'ñ', 'ó', 'ô', 'ö', 'ø', 'ù', 'û', 'ü', 'ā', 'ă', 'ć', 'č', 'ē', 'ě', 'ī', 'ō', 'ť', 'ū', '˚', 'τ', 'ย', 'ร', 'อ', '่', '€', '∇', '♪', '♫', '你', '葱', '送', '–', '—', '‘', '’', '“', '”','0', '1', '2', '3', '4', '5', '6', '7', '8', '9','#', '$', '%', '&', '!', '"', "'", '(', ')', '*', ':', ';','…']
# rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
# for document in doc.findall('//content'):
#     i +=1
#     # get each talk
#     corpus = document.text.lower()
#     # remove unwanted characters
#     corpus = re.sub(rx, '', corpus)
#     # create total corpus
#     totalcorpus = totalcorpus + " S " + corpus + " E "
# print(len(totalcorpus))
# print(i)
# corpus = totalcorpus

# with open(FLAGS.data_path, "w") as text_file:
#     text_file.write(corpus) 


# ### Load text file if processed before

# In[115]:

def readfile(data_path):
    corpus = open(data_path, "r")
    corpus = corpus.read()
    return corpus


# ### Create character to index and index to character functions

# In[111]:

def get_dicts(corpus):
    chars = sorted(list(set(corpus)))
    char2ind = dict((c, i) for i, c in enumerate(chars))
    ind2char = dict((i, c) for i, c in enumerate(chars))
    return char2ind, ind2char

# In[122]:

# Split text into overlapping sentences with step size 3.
# print('Splitting text into sequences...')
def split2sentences(corpus, maxlen):
    sentencelen = maxlen+1
    step = 5
    sentences = []
    for i in range(0, len(corpus) - sentencelen, step):
        sentences.append(corpus[i: i + sentencelen])
    return sentences


# In[114]:

def vectorize(sentences, maxlen, charlen, char_indices):
    X = np.zeros((len(sentences), maxlen, charlen), dtype=np.bool)
    Y = np.zeros((len(sentences), maxlen, charlen), dtype=np.bool)
    
    # vectorize the entire set by splitting sentences into X and Y, where Y is X shifted
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            if t==0:
                X[i, t, char_indices[char]] = 1
            elif t==maxlen:
                Y[i, t-1, char_indices[char]] = 1
            else:
                X[i, t, char_indices[char]] = 1
                Y[i, t-1, char_indices[char]] = 1
    return X,Y

def unvectorize(tensor, ind2char):
    sentences = []
    print(tensor.shape)
    for i, sentence in enumerate(tensor):
        x = ""
        for j, char in enumerate(sentence):
            y = ind2char[np.argmax(char)]
            x += y
        sentences.append(x)
    return sentences


# In[136]:

def generate_data(FLAGS):
    corpus = readfile(FLAGS.data_path)
    char2ind, ind2char = get_dicts(corpus)
    sentences = split2sentences(corpus, FLAGS.maxlen)
    num_sent = len(sentences)
    batch_size = FLAGS.batch_size
    while 1:
        for j in range(0, num_sent, batch_size):
            batch = sentences[j:j+batch_size]
            X, Y = vectorize(batch, FLAGS.maxlen, len(char2ind), char2ind)
            yield (X, Y)
            


# In[146]:

FLAGS.samples_per_epoch = len(split2sentences(readfile(FLAGS.data_path), FLAGS.maxlen))//FLAGS.batch_size
print('sample per epoch: ' + str(FLAGS.samples_per_epoch))
char2ind, ind2char = get_dicts(corpus=readfile(FLAGS.data_path))
FLAGS.charlen = len(char2ind)


# ## Build model

# In[119]:

N_HIDDEN = 128
N_HIDDEN2 = 128
LEARNING_RATE = 0.01
DROPOUT = 0


# In[120]:

print('Building training model...')
model = Sequential()
# The output of the LSTM layer are the hidden states of the LSTM for every time step. 
model.add(GRU(N_HIDDEN, return_sequences = True, input_shape=(FLAGS.maxlen, FLAGS.charlen)))
model.add(Dropout(DROPOUT))
model.add(GRU(N_HIDDEN2, return_sequences = True))
model.add(Dropout(DROPOUT))
# Two things to notice here:
# 1. The Dense Layer is equivalent to nn.Linear(hiddenStateSize, hiddenLayerSize) in Torch.
#    In Keras, we often do not need to specify the input size of the layer because it gets inferred for us.
# 2. TimeDistributed applies the linear transformation from the Dense layer to every time step
#    of the output of the sequence produced by the LSTM.
model.add(TimeDistributed(Dense(N_HIDDEN2)))
model.add(TimeDistributed(Activation('relu'))) 
model.add(TimeDistributed(Dense(FLAGS.charlen)))  # Add another dense layer with the desired output size.
model.add(TimeDistributed(Activation('softmax')))
model.compile(loss='categorical_crossentropy', optimizer = RMSprop(lr=LEARNING_RATE, decay=0.03, clipnorm=5))
print(model.summary()) # Convenient function to see details about the network model.
model.load_weights('model2.h5')
# The only difference with the "training model" is that here the input sequence has 
# a length of one because we will predict character by character.
print('Building Inference model...')
inference_model = Sequential()
# Two differences here.
# 1. The inference model only takes one sample in the batch, and it always has sequence length 1.
# 2. The inference model is stateful, meaning it inputs the output hidden state ("its history state")
#    to the next batch input.
inference_model.add(GRU(N_HIDDEN, batch_input_shape=(1, 1, FLAGS.charlen), stateful = True, return_sequences = True))
inference_model.add(Dropout(DROPOUT))
inference_model.add(GRU(N_HIDDEN2, batch_input_shape=(1, 1, FLAGS.charlen), return_sequences = True))
inference_model.add(Dropout(DROPOUT))
# Since the above LSTM does not output sequences, we don't need TimeDistributed anymore.
inference_model.add(Dense(N_HIDDEN2))
inference_model.add(Activation('relu'))
inference_model.add(Dense(FLAGS.charlen))
inference_model.add(Activation('softmax'))

# In[145]:

epoch=0
print('Training model')
while 1:
    epoch += 1
    t0 = time.time()
    model.reset_states()
    hist = model.fit_generator(generate_data(FLAGS), nb_epoch=1, samples_per_epoch=FLAGS.samples_per_epoch)
    model.save_weights("model2.h5")
    t1 = time.time()
    total = t1-t0
    
    orig_stdout = sys.stdout
    f = open('out2.txt', 'a+')
    sys.stdout = f

    print("------------- EPOCH" + str(epoch) + " ----------------")
    print('Time taken: ')
    print(total)

    if epoch%10 == 0:
        # Copy the weights of the trained network. Both should have the same exact number of parameters (why?).
        inference_model.load_weights('model2.h5')
        inference_model.reset_states()
        # Given the start Character 'S' (one-hot encoded), predict the next most likely character.
        startChar = np.zeros((1, 1, FLAGS.charlen))
        startChar[0, 0, char2ind['S']] = 1
        text=""
        for i in range(1000):
            nextCharProbs = inference_model.predict(startChar)
            nextCharProbs = np.asarray(nextCharProbs).astype('float64') # Weird type cast issues if not doing this.
            nextCharProbs = nextCharProbs / nextCharProbs.sum()  # Re-normalize for float64 to make exactly 1.0.

            nextCharId = np.random.multinomial(1, nextCharProbs.squeeze(), 1).argmax()
            text += ind2char[nextCharId] # The comma at the end avoids printing a return line character.
            startChar.fill(0)
            startChar[0, 0, nextCharId] = 1
        print('Generated Text:')
        print(text)
        sys.stdout = orig_stdout
        f.close()

