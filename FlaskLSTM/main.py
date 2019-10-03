from flask import Flask, request, jsonify
import sys,os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier

import time
import json
#EDA Packages
import pandas as pd
import numpy as np

# ML Packages
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
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
    #Check JSON data
    if not request.is_json:
        return ErrorMSG

    content = request.get_json()
    text = content['Text']
    if text is None:
        return ErrorMSG

    # Link to dataset from github
    model = load_model('lstm_sent.h5')

    vocabs = np.load('vocabs.npy').item()

    example="I can't properly express how pleased I am that this show turned out as well as it did. It's is extremely well shot, well written, and well acted."

    x = np.array([[vocabs[word] if word in vocabs.keys() else 0 for word in seq.split()] for seq in [text]])

    print ("Example Sentence: ", x)

    x_pad=sequence.pad_sequences(x,maxlen=100, padding='post')

    print ("example class(1 means positive, 0 means negative): ", (model.predict_classes(x_pad)))

    print ("example probability: ", model.predict_proba(x_pad))

    start = time.clock()

    my_class = model.predict_classes(x_pad)

    my_prob =model.predict_proba(x_pad)

    end = time.clock()

    print(end - start)

    myJSON = {}

    classed =my_class.tolist()[0]

    prob =my_prob.tolist()

    myJSON['text'] = text

    myJSON['prediction'] = str(classed)

    myJSON['Probability'] = str(prob)


    myJSON['process_time'] = str(end - start)

    return jsonify(myJSON)


if __name__ == '__main__':
    app.run(host="127.0.0.1",port=8080,debug=True, threaded=False)

