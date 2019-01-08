#!/usr/bin/env python
# -*- coding: utf-8 -*-
# All copyright reserved by Chen Min
# Author  : milk9815100@gmail.com
# Date    : 12/20/18

import logging
import argparse
import warnings

from experiment import Experiment

def config_logger():
    # create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(levelname)s:%(asctime)s: - %(name)s - : %(message)s')
    ch.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(ch)
    
    
def main(data_parameters, model_parameters):
    warnings.filterwarnings("ignore")
    Experiment(data_parameters, model_parameters)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_name', type=str, default="adult")
    parser.add_argument('--target_size', type=int, default=10000)
    parser.add_argument('--shadow_size', type=int, default=10000)
    parser.add_argument('--train_ratio', type=float, default=0.5)
    
    parser.add_argument('--target_classifier_type', type=str, default="svm")
    parser.add_argument('--attack_classifier_type', type=str, default="svm")
    parser.add_argument('--num_shadows', type=int, default=5)
    parser.add_argument('--evaluation_metric', type=str, default="accuracy")

    args = parser.parse_args()
    
    data_paramters = {
        "dataset_name": args.dataset_name,
        "target_size": args.target_size,
        "shadow_size": args.shadow_size,
        "train_ratio": args.train_ratio
    }
    model_parameters = {
        "target_classifier_type": args.target_classifier_type,
        "attack_classifier_type": args.attack_classifier_type,
        "num_shadows": args.num_shadows,
        "evaluation_metric": args.evaluation_metric
    }
    
    config_logger()
    main(data_paramters, model_parameters)
