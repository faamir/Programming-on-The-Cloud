import numpy as np
from keras.preprocessing import sequence
from keras.models import load_model

#load the model
model = load_model('lstm_sent.h5')
vocabs = np.load('vocabs.npy').item()

example="I can't properly express how pleased I am that this show turned out as well as it did. It's is extremely well shot, well written, and well acted."
x = np.array([[vocabs[word] if word in vocabs.keys() else 0 for word in seq.split()] for seq in [example]])
print ("Example Sentence: ", x)
x_pad=sequence.pad_sequences(x,maxlen=100, padding='post')
print ("example class(1 means positive, 0 means negative): ", (model.predict_classes(x_pad)))
print ("example probability: ", model.predict_proba(x_pad))

