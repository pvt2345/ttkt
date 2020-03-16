from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers import Embedding, Dense, Flatten, Dropout
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import argparse

#ap = argparse.ArgumentParser()
#ap.add_argument("-p", "--prebuilt", required=True,
#                help="Path to pre-built data")
#ap.add_argument("-w", "--weights", required=True, help="Path to model file")
#ap.add_argument("-o", "--output", required=True, help="Path to output image")
#args = vars(ap.parse_args())

print("[INFO] loading prebuilt-data...")

# load tokenizer
#pathToTokenizer = os.path.join(args['prebuilt'], 'tokenizer.pickle')
pathToTokenizer = 'D:/OneDrive/Works/ttkt/pretrained/tokenizer.pickle'
with open(pathToTokenizer, 'rb') as handle:
    tokenizer = pickle.load(handle)
    
# load labels
#pathToLabels = os.path.join(args['prebuilt'], 'labels.npy')
pathToLabels = 'D:/OneDrive/Works/ttkt/pretrained/labels.npy'
labels = np.load(pathToLabels)

# load data
#pathToData = os.path.join(args['prebuilt'], 'vntc.npy') 
pathToData = 'D:/OneDrive/Works/ttkt/pretrained/vntc.npy'
data = np.load(pathToData)

print("[INFO] shuffling data...")
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

lb = LabelBinarizer().fit(labels)
labels = lb.transform(labels)


X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.25, random_state=42)

maxlen = 500
max_words = 10000

print("[INFO] building and compiling model...")
model = Sequential()
model.add(Embedding(max_words, 10, input_length=maxlen))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# opt = SGD(lr=0.1)
model.compile(optimizer='rmsprop', loss="categorical_crossentropy",
              metrics=["accuracy"])


#checkpoint = ModelCheckpoint(
#   args["weights"], monitor="val_loss", save_best_only=True, verbose=1)
checkpoint = ModelCheckpoint(
   'weights.hdf5', monitor="val_loss", save_best_only=True, verbose=1)
#
#checkpoint = ModelCheckpoint(
#          'weights.hdf5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
callbacks = [checkpoint]

print("[INFO] training model...")
n_epochs = 15
his = model.fit(X_train, y_train, validation_data=(
        X_test, y_test), batch_size=64, epochs=n_epochs, callbacks=callbacks)

predictions = model.predict(X_test)
print(classification_report(y_test.argmax(axis=1),
                            predictions.argmax(axis=1), target_names=lb.classes_))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, n_epochs), his.history["loss"], label="train_loss")
plt.plot(np.arange(0, n_epochs), his.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, n_epochs), his.history["acc"], label="train_acc")
plt.plot(np.arange(0, n_epochs), his.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()

#plt.savefig(args['output'])
