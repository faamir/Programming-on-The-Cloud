from flask import Flask, request, jsonify

import sys,os


import time

import json

#EDA Packages

import pandas as pd

import numpy as np

 

# ML Packages

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB
import pickle

from sklearn.externals import joblib

# ML Packages

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier


from keras.preprocessing import sequence

from keras.models import load_model

 

app = Flask(__name__)

 

ErrorMSG ="""

Please sent json object as

{

"Text":"Hi this is first posted text"

}

"""

 

@app.route('/ML', methods = ['POST'])

def postJsonHandler():

    #print (request.is_json)

    #Check JSON data

    if not request.is_json:

        return ErrorMSG

    content = request.get_json()

    text = content['Text']

    if text is None:

        return ErrorMSG

    # Link to dataset from github

    url= "data.csv"

    df= pd.read_csv(url)

    df_data = df[["Summary","Sentiment"]]

    # Features and Labels

    df_x = df_data['Summary']

    df_y = df_data.Sentiment

    # Extract Feature With CountVectorizer

    corpus = df_x

    cv = CountVectorizer()

    X = cv.fit_transform(corpus) # Fit the Data

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size=0.33, random_state=42)

    #Naive Bayes Classifier

    

    from sklearn.naive_bayes import MultinomialNB

    clf = MultinomialNB()

    clf.fit(X_train,y_train)

    clf.score(X_test,y_test)

    start = time.clock()

    data = [text]

    vect = cv.transform(data)

    my_prediction = clf.predict(vect)

    end = time.clock()

    print(end - start)

    myJSON = {}

    myJSON['text'] = text

    myJSON['prediction'] = str(my_prediction[0])

    myJSON['process_time'] = str(end - start)

    #print(myJSON)

    return jsonify(myJSON)

    #return "OK"

 

@app.route('/')

@app.route('/home')

def index():

    return "<h1>Classification on the Cloud</h1><br/>"

 

@app.route('/DeepLearning', methods = ['POST'])

def postJson_v2Handler():

    #print (request.is_json)

    #Check JSON data

    if not request.is_json:

        return ErrorMSG

    content = request.get_json()

    text = content['Text']

    if text is None:

        return ErrorMSG

    # Link to dataset from github

    url= "data.csv"
    
    df= pd.read_csv(url)
    
    df_data = df[["Summary","Sentiment"]]

    # Features and Labels

    df_x = df_data['Summary']

    df_y = df_data.Sentiment

    # Extract Feature With CountVectorizer

    corpus = df_x

    cv = CountVectorizer()

    X = cv.fit_transform(corpus) # Fit the Data

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, df_y, test_size=0.33, random_state=42)

    #Naive Bayes Classifier

    

    from sklearn.naive_bayes import MultinomialNB

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(50, 2), random_state=1)

    clf.fit(X_train,y_train)

    clf.score(X_test,y_test)

    start = time.clock()

    data = [text]

    vect = cv.transform(data)

    my_prediction = clf.predict(vect)
    
    my_prob = clf.predict_proba(vect)

    end = time.clock()

    print(end - start)

    myJSON = {}

    myJSON['text'] = text

    myJSON['prediction'] = str(my_prediction[0])
    
    myJSON['probability'] = str(my_prob[0])

    myJSON['process_time'] = str(end - start)

    #print(myJSON)

    return jsonify(myJSON)

    #return "OK"





if __name__ == '__main__':

    app.run(host="127.0.0.1",port=8080,debug=True,threaded=False)