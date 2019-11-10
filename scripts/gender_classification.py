#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 15:50:30 2019

@author: Bharath Sankaran
@Copyright: Copyright (c) 2019, Scaled Robotics, SL., All rights reserved.
@filename: gender_classification.py

Redistribution prohibited without prior written consent

"""
import os
import tensorflow as tf

import shutil

import read_data as rd
import matplotlib.pyplot as plt

tf.reset_default_graph()

dataset_path = '/data/scaled/tensorflow_datasets/'
model_dir = 'kernel_log'
    
class TrainLinearModel(object):
    
    def __init__(self, X_train, Y_train, X_validate, Y_validate, 
                       modeldir = model_dir):        
        self.model_dir =  modeldir
        
        """ Get current working directory """
        curr_dir = os.getcwd()
        """ remove model directory """
        remove_dir = os.path.join(curr_dir, model_dir)
        try:
            shutil.rmtree(remove_dir)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror) )
            
        
        fc = rd.FeatureClass()
        print('Computing features on training set...')
        self.X_train = fc.extract_basic_features(X_train)        
        self.Y_train  = Y_train.copy()
        self.Y_train[Y_train == -1] = 0
        
        print('Computing features on testing set...')
        self.X_validate = fc.extract_basic_features(X_validate)        
        self.Y_validate  = Y_validate.copy()
        self.Y_validate[Y_validate == -1] = 0
        self.dimension = self.X_train.shape[1]
        
        self.init_linear_model()
        
        
    def init_linear_model(self):
        
        """ Train linear model """
        print('Setting up linear model...')
        feat_column = tf.contrib.layers.real_valued_column('features', 
                                                           dimension
                                                           = self.dimension)
        """ Building a classifier """
        """ Your learning algorithm would probably go here """
        self.estimator = None 
        
        self.train_input_fn = tf.estimator.inputs.numpy_input_fn(
                {"features": self.X_train},    
                y=self.Y_train,
                batch_size=200,    
                num_epochs=None,    
                shuffle=True)	
        
    def train(self):        
        self.estimator.train(input_fn=self.train_input_fn, steps=1000)
        
    def validate(self):
        
        test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"features": self.X_validate},
        y=self.Y_validate,
        batch_size=self.X_validate.shape[0],
        num_epochs=1,
        shuffle=False)
        
        self.estimator.evaluate(input_fn=test_input_fn, steps=1)
        
        
def main():
    
    """ First downlaod the images and associated labels"""
    rd.download_extract(dataset_path)
    
    """ Create the dataset class to extract the data """
    """ This class will extract the data form the downloaded class 
    with the labels for the specific classification test """
    dataset = rd.Dataset(dataset_path)
    
    """ Verify if your files have loaded properly """
    """ Display a grid of images """
    
    data_batch, indices = dataset.get_image_batch(12)
    test_image = rd.Dataset.images_display_grid(data_batch, 'RGB')
    plt.imshow(test_image)
    plt.imsave(os.path.join(dataset_path, 'test.jpg'), test_image)
    
    """ Now train your model """
    """ This is the bit where you need to code on your own """
    """ I have provided a starter script that computes features using the 
    feature class and trains a basic linear discriminative model 
    (logistic regression) """
    linear_model = TrainLinearModel(dataset.X_train, dataset.Y_train, 
                                    dataset.X_validate, dataset.Y_validate)
    
    print('Training linear model...')
    linear_model.train()
    print('Testing linear model...')
    linear_model.validate()
    
    
if __name__ == "__main__":
    main()   
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
