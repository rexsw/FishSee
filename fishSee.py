# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os
import cv2
import keras
import matplotlib.pylab as plt
from sklearn import manifold

tf.logging.set_verbosity(tf.logging.INFO)
tf.set_random_seed(13)
np.random.seed(13)

def morpholog(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(list(gray))

    ret,binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)

    binary = cv2.bitwise_not(binary)

    kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(3,3))

    roi = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel,iterations = 1)
    return roi

def load_fish():
    listing = os.listdir("fishRecognition_GT/fish_image")
    train_labels = []
    train_data = []
    eval_labels = []
    eval_data = []
    for i,subdir in enumerate(listing):
        images = os.listdir("fishRecognition_GT/fish_image/"+subdir)
        for image in images:
            if np.random.randint(0,11) < 2:
                eval_labels.append(i)
#                mask = cv2.resize(cv2.imread("fishRecognition_GT/mask_image/mask"+subdir[4:]+"/mask"+image[4:]),(130,130))
#                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
#                mask = np.reshape(mask, (130,130,1))
                im = cv2.resize(cv2.imread("fishRecognition_GT/fish_image/"+subdir+"/"+image),(130,130))
                mask = morpholog(im)
                mask = np.reshape(mask, (130,130,1))
                im = np.concatenate((im, mask),axis=2)
                eval_data.append(im)
            else:
                train_labels.append(i)
#                mask = cv2.resize(cv2.imread("fishRecognition_GT/mask_image/mask"+subdir[4:]+"/mask"+image[4:]),(130,130))
#                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
#                mask = np.reshape(mask, (130,130,1))
                im = cv2.resize(cv2.imread("fishRecognition_GT/fish_image/"+subdir+"/"+image),(130,130))
                mask = morpholog(im)
                mask = np.reshape(mask, (130,130,1))
                im = np.concatenate((im, mask),axis=2)
                train_data.append(im)
    return np.array(train_data/np.float32(255),dtype=np.float32), np.array(train_labels), np.array(eval_data/np.float32(255),dtype=np.float32), np.array(eval_labels)

fishid = ["Dascyllus reticulatus","Plectroglyphidodon dickii","Chromis chrysura"
          "Amphiprion clarkii","Chaetodon lunulatus","Chaetodon trifascialis",
          "Myripristis kuntee", "Acanthurus nigrofuscus","Hemigymnus fasciatus",
          "Neoniphon sammara","Abudefduf vaigiensis","Canthigaster valentini",
          "Pomacentrus moluccensis","Zebrasoma scopas","Hemigymnus melapterus",
          "Lutjanus fulvus","Scolopsis bilineata","Scaridae","Pempheris vanicolensis",
          "Zanclus cornutus","Neoglyphidodon nigroris","Balistapus undulatus",
          "Siganus fuscescens"]


train_data, train_labels,eval_data, eval_labels = load_fish()
img_x, img_y = 130, 130
train_cats = train_labels
eval_cats = eval_labels

classes = max(train_labels) + 1
train_labels = keras.utils.to_categorical(train_labels, num_classes=classes)
eval_labels = keras.utils.to_categorical(eval_labels, num_classes=classes)

input_shape = (img_x, img_y, 3)
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
model.add(keras.layers.MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))
model.add(keras.layers.Conv2D(64, (5, 5), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(5, 5)))
model.add(keras.layers.Conv2D(32, (5, 5), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(5, 5)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(1000, activation='relu'))
model.add(keras.layers.Dropout(0.8))
model.add(keras.layers.Dense(1000, activation='relu'))
model.add(keras.layers.Dense(classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01),
              metrics=['accuracy'])
class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))
history = AccuracyHistory()
model.fit(train_data, train_labels,
          batch_size=128,
          epochs=40,
          verbose=1,
          validation_data=(eval_data, eval_labels),
          callbacks=[history])
score = model.evaluate(eval_data, eval_labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

plt.plot(history.acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()
#
model = keras.models.Sequential()
model.add(keras.layers.Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
model.add(keras.layers.MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))
model.add(keras.layers.Conv2D(64, (5, 5), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(5, 5)))
model.add(keras.layers.Conv2D(32, (5, 5), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(5, 5)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(1000, activation='relu'))
model.add(keras.layers.Dropout(0.8))
model.add(keras.layers.Dense(2000, activation='relu'))
model.add(keras.layers.Dropout(0.8))
model.add(keras.layers.Dense(1000, activation='relu'))
model.add(keras.layers.Dense(classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01),
              metrics=['accuracy'])

history = AccuracyHistory()
model.fit(train_data, train_labels,
          batch_size=128,
          epochs=40,
          verbose=1,
          validation_data=(eval_data, eval_labels),
          callbacks=[history])
score = model.evaluate(eval_data, eval_labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

plt.plot(history.acc)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()
#colours = ['r','b','g','y']
#c = {}
#train_data = np.reshape(train_data,(train_data.shape[0], train_data.shape[1]*train_data.shape[2]*train_data.shape[3]))
#for i, key in enumerate(train_cats):
#    c[frozenset(train_data[i].tolist())] = colours[key]
#n_neighbors = 5
#np.random.shuffle(train_data)
#train_data = train_data[:100]
#mana_data = []
#mana_colour = []
#for i in train_data:
#    mana_colour.append(c[frozenset(i.tolist())])
#    mana_data.append(i)
#trans_data = manifold.LocallyLinearEmbedding(n_neighbors, 2).fit_transform(mana_data).T
#plt.scatter(trans_data[0], trans_data[1],c=mana_colour)
#plt.show()