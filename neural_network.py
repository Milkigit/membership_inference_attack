#!/usr/bin/env python
# -*- coding: utf-8 -*-
# All copyright reserved by Chen Min
# Author  : milk9815100@gmail.com
# Date    : 12/20/18

import logging
import warnings

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten

import config
#from classifier import Records


class NeuralNetwork:
    def __init__(self, records):
        self.logger = logging.getLogger("nn")
        
        self.model = Sequential()
        self.records = records

    def reformat_image(self, image_rows, image_columns):
        self.logger.info("reformat mnist dataset")
        
        if K.image_data_format() == 'channels_first':
            self.records.train_x = self.records.train_x.reshape(self.records.train_x.shape[0], 1, image_rows, image_columns)
            self.records.test_x = self.records.test_x.reshape(self.records.test_x.shape[0], 1, image_rows, image_columns)
            self.input_shape = (1, image_rows, image_columns)
        else:
            self.records.train_x = self.records.train_x.reshape(self.records.train_x.shape[0], image_rows, image_columns, 1)
            self.records.test_x = self.records.test_x.reshape(self.records.test_x.shape[0], image_rows, image_columns, 1)
            self.input_shape = (image_rows, image_columns, 1)

        self.records.train_x = self.records.train_x.astype('float32')
        self.records.test_x = self.records.test_x.astype('float32')
        self.records.train_x /= 255
        self.records.test_x /= 255

        self.records.train_y = keras.utils.to_categorical(self.records.train_y, 10)
        self.records.test_y = keras.utils.to_categorical(self.records.test_y, 10)

    def construct_network(self):
        self.logger.info("constructing neural network")
        
        self.model.add(Conv2D(32, kernel_size=(3, 3),
                              activation='relu',
                              input_shape=self.input_shape))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(10, activation='softmax'))

        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=keras.optimizers.Adadelta(),
                           metrics=['accuracy'])
        
    def fit(self, batch_size=128, epochs=12):
        self.logger.info("training neural network")
        
        self.model.fit(self.records.train_x, self.records.train_y,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=2)

    def predict_proba(self, x):
        return self.model.predict(x)
    
    def predict(self, x):
        prob_predict = self.model.predict(x)
        
        return np.argmax(prob_predict, axis=1)
    
    def evaluation(self, true_y, pred_y):
        return accuracy_score(true_y, pred_y)
        
        
def main():
    warnings.filterwarnings("ignore")
    output_file = None
    logging.basicConfig(filename=output_file,
                        format='%(levelname)s:%(asctime)s: - %(name)s - : %(message)s',
                        level=logging.DEBUG)

    logging.info("loading data")

    load_data = np.loadtxt(config.DATASET_PATH + "mnist.csv", dtype=np.uint16, delimiter=",")
    features = load_data[:, 1:]
    labels = load_data[:, 0]

    records = Records()
    records.train_x, records.test_x, records.train_y, records.test_y = train_test_split(features, labels, train_size=0.5)
    
    nn = NeuralNetwork(records)
    nn.reformat_image(28, 28)
    nn.construct_network()
    nn.fit(epochs=1)
    pred_y = nn.predict(records.test_x)
    true_y = np.argmax(records.test_y, axis=1)
    accuracy = nn.evaluation(true_y, pred_y)
    
    logging.info("the accuracy score is %s" % (accuracy,))


if __name__ == "__main__":
    main()