#LSTM Implementation (Deep Learning) for Classification

#!pip install gensim
#!conda install --yes theano
#!conda install --yes tensorflow
#!conda install --yes keras
import pandas as pd
import numpy as np
import time
import os
import nltk
from nltk.corpus import stopwords
import re
from collections import Counter
import itertools
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import LSTM, GRU
from keras.preprocessing import sequence
from keras.layers import Dense, Dropout, Embedding
from keras.utils.np_utils import to_categorical
from keras import optimizers
import keras_metrics
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from helper_functions import *
import matplotlib.pyplot as plt
#nltk.download("stopwords")
#nltk.download("wordnet")



#load csv dataset
dataset = pd.read_csv('./data/largeimdb.csv', encoding='latin1')
print(dataset.head())




#get sentences, remove numbers, get word frequency, tokenize it and merge similar words
similarity_word_reduce = WordNetLemmatizer()
def get_ds(dataset):
    datasetx =dataset
    datasetx["Summary"] = datasetx.apply(remove_numerics, axis=1)
    datasetx["tokenize"] = datasetx.apply(word_frequncy_maker, axis=1)
    #print ("dataset head in received", datasetx.head())
    n = []
    for i in datasetx["tokenize"]:
        list = []
        for j in i.split(","):
            list.append(similarity_word_reduce.lemmatize(j))
        n.append(" ".join(list))
    datasetx['Summary_lem'] = n
    """
    stop_words = set(stopwords.words('english')) 
    sentences = [] 
      
    for w in datasetx['Summary_lem']: 
        if w not in stop_words: 
            sentences.append(w)
    datasetx['Summary_lem'] = sentences
    """
    #print ("dataset head in received", datasetx.head())
    return datasetx['Summary_lem']

dataset['Summary'] = get_ds(dataset[['Summary']])


# Cut sentences in dataset
list_all=[]
sentence_len = 100
for i in dataset.values:
    if len(i[0]) <= sentence_len and len(i[0]) >= 5:
        list_all.append([i[0], i[1]])
        #print (len(i[0]))
    elif len(i[0]) > sentence_len:
        tmp=' '.join(i[0].split()[0:sentence_len])
        list_all.append([tmp,i[1]])


#make our dataset again
dataset=pd.DataFrame(list_all,columns=['Summary','Sentiment'])
#print("datasetset df, length  : \n", dataset.head())
#print(dataset)
print("number of positive sentences: ", len(dataset[dataset['Sentiment']==1]),
      " ,number of negative sentences: ", len(dataset[dataset['Sentiment']==0]))


#load x and labels
def load_x_y():
    # Load dataset from files
    # Split by words
    x_all = dataset["Summary"].values
    x_all = [string_cleaner(sent) for sent in x_all]
    #print("x_test from 0 to 1 in load_x_y func: ", x_all[0:2]) #simply 2 sentences from datasetset which cleaned
    x_all = [s.split(" ") for s in x_all]
    y_all=dataset['Sentiment'].values
    
    return [x_all, y_all]


#pad all sentences in case all sentences have the same length
sentences_length =  sentence_len
def padding(sequences, pad="<PWD/>"):
    padded_sentences = []
    for i in range(len(sequences)):
        sentence = sequences[i]
        pad_no = sentences_length - len(sentence)
        if pad_no<=0:
            tmp=sentence[0:sentences_length-1]
            tmp.append(pad)
            new_sequence=' '.join(tmp).split()
        else:
            new_sequence = sentence + [pad] * pad_no
            
        padded_sentences.append(new_sequence)
    #print ("padded_sentences and length of it in padding func",
                #padded_sentences[0],len(padded_sentences[0])) 
    return padded_sentences


#build vocabulary, word2index and index2word
def vocab_builder(sequences):
    
    word_cnt = Counter(itertools.chain(*sequences))
    vocab_invariant = [x[0] for x in word_cnt.most_common()]
    vocabs = {x: i for i, x in enumerate(vocab_invariant)}
    #print("vocabs", vocabs)
    #print("vocab_invariant", vocab_invariant)
    return [vocabs, vocab_invariant]


#map sentences
def mapper(sequences, labels, vocabs):
    
    x = np.array([[vocabs[str(word)] for word in sentence] for sentence in sequences])
    y = np.array(labels)
    return [x, y]


#load x, labels, vocab and vocab invariants for final processing
def dataset_loader():
    sequences, labels = load_x_y()
    #print("x sequences 0 to 1 raw: ", sequences[0:2]) 
    sentences_padded = padding(sequences)
    #print("x sequences 0 to 1 after padding (sequence length): ", sequences_padded[0:2]) 
    vocabs, vocab_invariant = vocab_builder(sentences_padded)
    x, y = mapper(sentences_padded, labels, vocabs)
    return [x, y, vocabs, vocab_invariant]

start = time.time()
print("Loading...")
x, y, vocabs, vocab_invariant = dataset_loader()
saved_vocabs = np.save('vocabs.npy', vocabs) 
print ("x: ", x)
#print("vocab_invariant: ", vocab_invariant) #all sentences in one list: ['<PWD/>', 'film', ....
print("vocabs size: {:d}".format(len(vocabs))) #all sentences in one dictionary: {'<PWD/>': 0, 'film': 1, ....
#print("vocabs : ", vocabs)
print("x.shape: ", x.shape) 
print("x.dtype: ", x.dtype) 
print ("print x shape before split in test valid train: ", x.shape) 


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)
shuffling = np.random.permutation(np.arange(len(y_train)))
x_shuffled = X_train[shuffling]
y_shuffled = y_train[shuffling]
shuffling_test = np.random.permutation(np.arange(len(y_test)))
x_shuffled_test = X_test[shuffling_test]
y_shuffled_test = y_test[shuffling_test]
print("x_shuffled train & validation shape: ", x_shuffled.shape)
print("y_shuffled train & validation shape : ", y_shuffled.shape)
print("x_shuffled_test shape: ", x_shuffled_test.shape)
print("y_shuffled_test shape: ", y_shuffled_test.shape)
#print(x_shuffled[0])
#print(y_shuffled[0])
y_shuffled=to_categorical(y_shuffled)
y_shuffled_test=to_categorical(y_shuffled_test)

RMSprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
Adagrad = optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
Adadelta = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
Adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
Adamax = optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
Nadam = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)


#LSTM Implementation
words = int(len(vocabs))
embedding_length = 32
model = Sequential()
model.add(Embedding(words, embedding_length, input_length=sentence_len))
model.add(Dropout(0.2))
#model.add(LSTM(64, return_sequences=True))
#model.add(LSTM(64, return_sequences=True))
model.add(LSTM(200))
model.add(Dropout(0.2))
model.add(Dense(2, activation='sigmoid'))
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy', keras_metrics.precision(), keras_metrics.recall()])
#model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=[keras_metrics.precision(), keras_metrics.recall()])
print(model.summary())


#Train phase
hist = model.fit(x_shuffled, y_shuffled, epochs=5, validation_split=0.15,batch_size=64)



#Testing phase
accuracy = model.evaluate(x_shuffled_test, y_shuffled_test, verbose=2, batch_size=64)
#print("Test Score: ", (accuracy[0]*100))
print("Test Accuracy: ", (accuracy[1]*100))
end = time.time()
print("Time elapsed (seconds): ", end - start)

print(hist.history.keys())
# plot Accuracy
plt.plot(hist.history['acc'], 'r')
plt.plot(hist.history['val_acc'], 'b')
plt.title('Accuracy of the model')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
'''
# plot loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Loss of the model')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
'''
'''
# plot Precision & Recall
plt.plot(hist.history['precision'], 'r')
plt.plot(hist.history['recall'], 'b')
plt.title(' Precision & Recall ')
plt.ylabel('%')
plt.xlabel('epoch')
plt.legend(['Precision', 'Recall'], loc='upper left')
plt.show()
'''

#test your own sentence
example="I can't properly express how pleased I am that this show turned out as well as it did. It's is extremely well shot, well written, and well acted."
x = np.array([[vocabs[word] if word in vocabs.keys() else 0 for word in seq.split()] for seq in [example]])
print ("Example Sentence: ", x)
x_pad=sequence.pad_sequences(x,maxlen=sentence_len, padding='post')
print ("example class(1 means positive, 0 means negative): ", (model.predict_classes(x_pad)))
print ("example probability: ", model.predict_proba(x_pad))
#save the model
model.save('lstm_sent.h5')
