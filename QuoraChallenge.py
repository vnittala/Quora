import numpy as np
import os
import pandas as pd
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Convolution1D, MaxPooling1D, BatchNormalization, Activation, Merge
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

# Setting variables
trainFile = "files/train.csv"
#trainFile = "files/train.csv"
GLOVE_DIR = "glove.6B/"

# Model parameters
epochs = 1
learning_rate = 0.05
decay_rate = learning_rate / epochs
momentum = 0.8
MAX_SEQUENCE_LENGTH = 40
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 300

# setting stopwords and stemmers
stemmer = SnowballStemmer('english')
stop = stopwords.words('english')

def clean_question(val):
    # Keep only spaces and characters
    # Remove all stop words
    #regex = re.compile("([^\s\w]|_)+") # This keeps numbers as well, in certain cases this doesn't help

    regex = re.compile("[^\sa-zA-Z]+")
    sentence = regex.sub("", str(val)).lower()
    sentence = sentence.split(" ")

    for word in list(sentence):
        if word in stop:
            sentence.remove(word)

    sentence = " ".join(sentence)
    return sentence


def word_stemming(word_text):

    sentence = []
    for cur_word in list(str(word_text).lower().split()):
        sentence.append(stemmer.stem(cur_word))

    sentence = " ".join(sentence)
    return sentence


def clean_dataframe(data):
    #data = data.dropna(how="any")

    for col in ['question1', 'question2']:
        data[col] = data[col].apply(clean_question)

    return data


def update_dataframe(data):
    # FS1 - First Feature Set

    data['q1stemmer'] = data.question1.apply(word_stemming)
    data['q2stemmer'] = data.question2.apply(word_stemming)
    #data['q1_count'] = data.question1.apply(lambda x: len(set(str(x).lower().split())))
    #data['q2_count'] = data.question2.apply(lambda x: len(set(str(x).lower().split())))
    #data['count_diff'] = abs(data.q1_count - data.q2_count)
    #data['common_count'] = data.apply(lambda x: len(set(str(x['question1']).lower().split()).intersection(set(str(x['question2']).lower().split()))), axis=1)

    cols = ['id', 'qid1', 'qid2', 'question1', 'question2']
    data.drop(cols, inplace=True, axis=1)
    # FS2 - Second Feature Set

    return data

# Main Program

# Process Train File
train = pd.read_csv(trainFile)
train = clean_dataframe(train)
train = update_dataframe(train)

# Preparing for NN execution
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(train.q1stemmer.values + train.q2stemmer.values.astype(str))

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

# output vector to compare
#output = np_utils.to_categorical(train.is_duplicate.values)
output = train.is_duplicate.values

x1 = tokenizer.texts_to_sequences(train.q1stemmer.values)
x1 = sequence.pad_sequences(x1, maxlen=MAX_SEQUENCE_LENGTH)

print('Shape of x1 data tensor:', x1.shape)

x2 = tokenizer.texts_to_sequences(train.q2stemmer.values.astype(str))
x2 = sequence.pad_sequences(x1, maxlen=MAX_SEQUENCE_LENGTH)

print('Shape of x2 data tensor:', x2.shape)

## Create Embedding Layer
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt'), encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

## Creating Embedding Matrix
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


embedding_filters = 64

print("Printing summary details.....\n")
model1 = Sequential()
model1.add(Embedding(len(word_index)+1, EMBEDDING_DIM,  weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False))
model1.add(Convolution1D(filters=embedding_filters, kernel_size=3, padding='same', activation='relu'))
model1.add(MaxPooling1D(pool_size=2))
model1.add(LSTM(EMBEDDING_DIM))
model1.add(Dense(EMBEDDING_DIM))
model1.add(Dropout(0.2))
model1.add(BatchNormalization())

print(model1.summary())

model2 = Sequential()
model2.add(Embedding(len(word_index)+1, EMBEDDING_DIM,  weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, trainable=False))
model2.add(Convolution1D(filters=embedding_filters, kernel_size=3, padding='same', activation='relu'))
model2.add(MaxPooling1D(pool_size=2))
model2.add(LSTM(EMBEDDING_DIM))
model2.add(Dense(EMBEDDING_DIM))
model2.add(Dropout(0.2))
model2.add(BatchNormalization())

print(model2.summary())

mergeModel = Sequential()
mergeModel.add(Merge([model1, model2], mode='concat'))

mergeModel.add(BatchNormalization())
mergeModel.add(Dense(EMBEDDING_DIM))
mergeModel.add(Dropout(0.2))
mergeModel.add(BatchNormalization())

mergeModel.add(Dense(1))
mergeModel.add(Activation('sigmoid'))

mergeModel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
mergeModel.fit([x1,x2], output, epochs=epochs, batch_size=128)

# Final evaluation of the model
#scores = model.evaluate(x2, output, verbose=0)
#print("Accuracy: %.2f%%" % (scores[1]*100))

# serialize model to JSON
model_json = mergeModel.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
mergeModel.save_weights("model.h5")
print("Saved model to disk")