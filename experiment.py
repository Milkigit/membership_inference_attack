#!/usr/bin/env python
# -*- coding: utf-8 -*-
# All copyright reserved by Chen Min
# Author  : milk9815100@gmail.com
# Date    : 12/20/18

import logging

import numpy as np

from attack import Attack
import config


class Experiment:
    def __init__(self, data_parameters, model_parameters):
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("experiment for dataset %s" % (data_parameters["dataset_name"],))
        
        self.load_data(data_parameters["dataset_name"])
        # split original data set into 2 parts, one part is for target model, the other part is for shadow models
        self.split_target_shadow(data_parameters["target_size"])
        
        attack = Attack(self.features, self.labels, model_parameters)
        attack.train_target_model(self.target_indices, data_parameters["train_ratio"])
        attack.train_shadow_model(self.shadow_indices, data_parameters["shadow_size"], data_parameters["train_ratio"])
        attack.train_attack_model()
        
        # self.error = attack.evaluation(model_parameters["evaluation_metric"])
        self.error = attack.evaluation()
        self.save_accuracy_data()

    def load_data(self, dataset_name):
        self.logger.info("loading data")
        
        load_data = np.loadtxt(config.DATASET_PATH + dataset_name + ".csv", dtype=np.uint16, delimiter=",")
        
        self.features = load_data[:, 1:]
        self.labels = load_data[:, 0]

    def split_target_shadow(self, target_size):
        indices = np.arange(self.labels.shape[0])
        
        self.target_indices = np.random.choice(indices, target_size, replace=False)
        self.shadow_indices = np.setdiff1d(indices, self.target_indices)
        
    def save_accuracy_data(self):
        pass