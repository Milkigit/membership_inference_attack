#!/usr/bin/env python
# -*- coding: utf-8 -*-
# All copyright reserved by Chen Min
# Author  : milk9815100@gmail.com
# Date    : 12/20/18

import logging
import csv
from collections import defaultdict

import numpy as np
from sklearn import cluster

import config


class Preprocess:
    def __init__(self):
        self.logger = logging.getLogger()
        
    def preprocess_adult(self):
        load_file = open(config.DATASET_PATH + "adult_original.csv", newline='')
        csv_data = csv.reader(load_file, dialect='excel')
        
        attribute_categories = [{} for _ in range(14)]
        attribute_categories_count = np.zeros(14, dtype=np.uint8)
        feature = np.zeros([45222, 14], dtype=np.uint8)
        label = np.zeros([45222, 1], dtype=np.uint8)
        
        for row_index, row in enumerate(csv_data):
            for attri_index, attri in enumerate(row):
                if attri_index == 14:
                    if attri == "<=50K":
                        label[row_index] = 0
                    else:
                        label[row_index] = 1
                else:
                    if attri not in attribute_categories[attri_index]:
                        attribute_categories[attri_index][attri] = attribute_categories_count[attri_index]
                        feature[row_index, attri_index] = attribute_categories[attri_index][attri]
                        attribute_categories_count[attri_index] += 1
                    else:
                        feature[row_index, attri_index] = attribute_categories[attri_index][attri]
                        
        save_data = np.concatenate((label, feature), axis=1)
        np.savetxt(config.DATASET_PATH + "adult.csv", save_data, delimiter=",", fmt='%i')
        
    def preprocess_purchase_binary(self, num_items):
        load_file = open("purchase.csv", 'r', newline='')
        csv_read = csv.reader(load_file, dialect='excel')
        items_count = defaultdict(str)
        items_selected = {}
        features = np.zeros([1, num_items], dtype=np.uint8)
        
        # count the number of each item
        for row_index, row in enumerate(csv_read):
            if row_index % 10000 == 0:
                self.logger.info("counting %s records" % (row_index, ))
                
            if row[5] in items_count:
                items_count[row[5]] += 1
            else:
                items_count[row[5]] = 1
                
        index = 0
        for item, count in sorted(items_count.items(), key=lambda item: item[1], reverse=True):
            if index < num_items:
                items_selected[item] = index
                index += 1
            else:
                break
            
        # process the dataset to binary features
        customer_id = ''
        feat = np.zeros([1, num_items], dtype=np.uint8)
        load_file.seek(0)
        for row_index, row in enumerate(csv_read):
            if row_index % 10000 == 0:
                self.logger.info("loading %s records" % (row_index, ))
                
            if row_index == 1:
                customer_id = row[0]
                feat = np.zeros([1, num_items], dtype=np.uint8)
            elif row_index > 1:
                if row[0] == customer_id:
                    if row[5] in items_selected:
                        feat[0, items_selected[row[5]]] = 1
                else:
                    features = np.concatenate((features, feat))
                    
                    feat = np.zeros([1, num_items], dtype=np.uint8)
                    if row[5] in items_selected:
                        feat[0, items_selected[row[5]]] = 1
                        
                    customer_id = row[0]
                    
        features = features[1:]
                    
        np.savetxt("feature.csv", features, delimiter=",", fmt='%i')
        
    def preprocess_purchase_cluster(self, n_clusters):
        self.logger.info("loading data")
        
        features = np.loadtxt(config.DATASET_PATH + "purchase_original.csv", dtype=np.uint8, delimiter=',')
        
        self.logger.info("clustering")
        
        model = cluster.KMeans(n_clusters=n_clusters)
        labels = model.fit_predict(features).reshape([features.shape[0], 1])
        
        self.logger.info("saving data")
        
        records = np.concatenate((labels, features), axis=1)
        
        np.savetxt(config.DATASET_PATH + "purchase_" + str(n_clusters) + ".csv", records, delimiter=",", fmt='%i')
        
        
def main():
    output_file = None
    logging.basicConfig(filename=output_file,
                        format='%(levelname)s:%(asctime)s: - %(name)s - : %(message)s',
                        level=logging.DEBUG)

    preprocess = Preprocess()

    # preprocess.preprocess_purchase_binary(10)
    preprocess.preprocess_purchase_cluster(2)
    
    
if __name__ == "__main__":
    main()



