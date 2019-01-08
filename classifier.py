#!/usr/bin/env python
# -*- coding: utf-8 -*-
# All copyright reserved by Chen Min
# Author  : milk9815100@gmail.com
# Date    : 12/20/18

import logging

from sklearn import linear_model, svm
from sklearn.metrics import accuracy_score
import numpy as np

from neural_network import NeuralNetwork


class Records:
    def __init__(self):
        self.train_x = None
        self.train_y = None
        self.test_x = None
        self.test_y = None


class Classifier:
    def __init__(self, records, classifier_type):
        self.logger = logging.getLogger(__name__)
        
        self.records = records
        self.classifier_type = classifier_type
        self.model = None
    
    def train_model(self):
        if self.classifier_type == 'linear':
            self.model = linear_model.LinearRegression()
        elif self.classifier_type == 'logistic':
            self.model = linear_model.LogisticRegression()
        elif self.classifier_type == 'svm':
            self.model = svm.SVC(probability=True, max_iter=1000)
        elif self.classifier_type == 'nn':
            nn = NeuralNetwork(self.records)
            nn.reformat_image(28, 28)
            nn.construct_network()
            nn.fit()
            self.model = nn
            
            return
        
        self.model.fit(self.records.train_x, self.records.train_y)
        
    def predict(self, x):
        return self.model.predict(x)
    
    def predict_probability(self, x):
        return self.model.predict_proba(x)
    
    def evaluation(self, true_y, pred_y):
        if self.classifier_type == "nn":
            true_y = np.argmax(true_y, axis=1)
            
        return accuracy_score(true_y, pred_y)
        