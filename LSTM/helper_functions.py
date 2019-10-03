#Helper Function for Text Cleaning

import pandas as pd
import numpy as np
import os
import nltk
from nltk.corpus import stopwords
import re
from collections import Counter
import itertools
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import LSTM
from keras.preprocessing import sequence
from keras.layers import Dense, Dropout, Embedding
from keras.utils.np_utils import to_categorical
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split


#String cleaning for dataset
def string_cleaner(sentences):
    
    sentences = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", sentences)
    sentences = re.sub(r",", " , ", sentences)
    sentences = re.sub(r"\?", "", sentences)
    sentences = re.sub(r"/", "", sentences)
    sentences = re.sub(r"\s{2,}", " ", sentences)
    sentences = re.sub(r"\'s", " \'s", sentences)
    sentences = re.sub(r"\'ve", " \'ve", sentences)
    sentences = re.sub(r"n\'t", " n\'t", sentences)
    sentences = re.sub(r"\'re", " \'re", sentences)
    sentences = re.sub(r"!", " ! ", sentences)
    sentences = re.sub(r"\(", "", sentences)
    sentences = re.sub(r"\)", "", sentences)
    sentences = re.sub(r"\'d", " \'d", sentences)
    sentences = re.sub(r"\'ll", " \'ll", sentences)
    #print (sentences)
    return sentences.strip().lower()



#count word frequency for dataset
word_frequency = {}
def word_frequncy_maker(col):
    datasetx =col["Summary"]
    #tokenize each word in dataset
    tokenize = nltk.wordpunct_tokenize(datasetx)
    tokenize_list =[]
    for tokenz in tokenize:
        tokenize_list.append(tokenz.lower())
        if tokenz.lower() in word_frequency:
            cnt = word_frequency[tokenz.lower()]
            cnt += cnt
            word_frequency[tokenz.lower()] = cnt
        else:
            word_frequency[tokenz.lower()] = 1
    return ",".join(tokenize_list)
            
            
            
#remove numbers from sentences
def remove_numerics(col):
    datasetx = col["Summary"]
    if type(datasetx) not in [int,float]:
        each_line = re.sub(r"[^A-Za-z\s]", " ", datasetx.strip())
        tokenize = each_line.split()
    else:
        tokenize = []
    return ' '.join(tokenize)