#!/usr/bin/env python
# -*- coding: utf-8 -*-
# All copyright reserved by Chen Min
# Author  : milk9815100@gmail.com
# Date    : 12/20/18

import logging

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

from classifier import Records, Classifier

class Attack:
    def __init__(self, features, labels, model_parameters):
        self.logger = logging.getLogger(__name__)
        
        self.features = features
        self.labels = labels
        self.target_classifier_type = model_parameters["target_classifier_type"]
        self.attack_classifier_type = model_parameters["attack_classifier_type"]
        self.num_shadows = model_parameters["num_shadows"]
    
    def train_target_model(self, target_indices, train_ratio):
        self.logger.info("training target model")
        
        records = Records()
        records.train_x, records.test_x, records.train_y, records.test_y = train_test_split(
            self.features[target_indices], self.labels[target_indices], train_size=train_ratio)
        
        classifier = Classifier(records, self.target_classifier_type)
        classifier.train_model()
        
        train_pred = classifier.predict(records.train_x)
        test_pred = classifier.predict(records.test_x)
        self.train_accuracy = classifier.evaluation(records.train_y, train_pred)
        self.test_accuracy = classifier.evaluation(records.test_y, test_pred)
        self.logger.info("train accuracy: %s" % (self.train_accuracy,))
        self.logger.info("test accuracy: %s" % (self.test_accuracy,))

        # x is feature, y is label
        attack_test_x_in = classifier.predict_probability(records.train_x)
        attack_text_x_out = classifier.predict_probability(records.test_x)
        attack_test_y_in = np.ones(records.train_y.shape[0])
        attack_test_y_out = np.zeros(records.test_y.shape[0])
        
        self.attack_test_x = np.concatenate((attack_test_x_in, attack_text_x_out))
        self.attack_test_y = np.concatenate((attack_test_y_in, attack_test_y_out))

        if self.target_classifier_type == "nn":
            records.train_y = np.argmax(records.train_y, axis=1)
            records.test_y = np.argmax(records.test_y, axis=1)
        self.classes_test = np.concatenate((records.train_y, records.test_y))
    
    def train_shadow_model(self, shadow_indices, shadow_size, train_ratio):
        num_classes = np.unique(self.labels).shape[0]
        self.attack_train_x = np.zeros([1, num_classes])
        self.attack_train_y = np.zeros(1)
        self.classes_train = np.zeros(1)
        
        for s in range(self.num_shadows):
            self.logger.info("training shadow model %s" % (s,))

            # choose shadow model training data set
            c_indices = np.random.choice(shadow_indices, shadow_size, replace=False)
            records = Records()
            records.train_x, records.test_x, records.train_y, records.test_y = train_test_split(
                self.features[c_indices], self.labels[c_indices], train_size=train_ratio)
    
            classifier = Classifier(records, self.target_classifier_type)
            classifier.train_model()
    
            attack_train_x_in = classifier.predict_probability(records.train_x)
            attack_train_x_out = classifier.predict_probability(records.test_x)
            attack_train_y_in = np.ones(records.train_y.shape[0])
            attack_train_y_out = np.zeros(records.test_y.shape[0])
    
            self.attack_train_x = np.concatenate((self.attack_train_x, attack_train_x_in, attack_train_x_out))
            self.attack_train_y = np.concatenate((self.attack_train_y, attack_train_y_in, attack_train_y_out))
            
            if self.target_classifier_type == "nn":
                records.train_y = np.argmax(records.train_y, axis=1)
                records.test_y = np.argmax(records.test_y, axis=1)
            self.classes_train = np.concatenate((self.classes_train, records.train_y, records.test_y))

        self.attack_train_x = self.attack_train_x[1:, :]
        self.attack_train_y = self.attack_train_y[1:]
        self.classes_train = self.classes_train[1:]
    
    def train_attack_model(self):
        train_indices = np.arange(self.attack_train_x.shape[0])
        test_indices = np.arange(self.attack_test_x.shape[0])
        unique_classes = np.unique(self.classes_train)
        
        self.true_y = np.zeros(1)
        self.pred_y = np.zeros(1)
        
        for c in unique_classes:
            self.logger.info("training attack model for class %s" % (int(c),))
            
            records = Records()

            c_train_indices = train_indices[self.classes_train == c]
            records.train_x, records.train_y = self.attack_train_x[c_train_indices], self.attack_train_y[c_train_indices]
            c_test_indices = test_indices[self.classes_test == c]
            records.test_x, records.test_y = self.attack_test_x[c_test_indices], self.attack_test_y[c_test_indices]

            classifier = Classifier(records, self.attack_classifier_type)
            classifier.train_model()
            c_pred_y = classifier.predict(records.test_x)
            
            self.true_y = np.concatenate((self.true_y, records.test_y))
            self.pred_y = np.concatenate((self.pred_y, c_pred_y))
            
        self.true_y = self.true_y[1:]
        self.pred_y = self.pred_y[1:]
    
    def evaluation(self):
        error = classification_report(self.true_y, self.pred_y)
        
        # if metric == "accuracy":
        #     error = accuracy_score(self.true_y, self.pred_y)
        # elif metric == "mis_rate":
        #     pass
        self.logger.info("train accuracy: %s" % (self.train_accuracy,))
        self.logger.info("test accuracy: %s" % (self.test_accuracy,))
        self.logger.info("attack accuracy: %s" % (accuracy_score(self.true_y, self.pred_y)))
        
        return error
