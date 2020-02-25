#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 20:41:35 2020

@author: linda
"""

import network

import read_data

training_data, validation_data, test_data= read_data.load_data_wrapper()
training_data=list(training_data)
#training_data.shape

net=network.Network([784,50,10])

net.SGD(training_data, 30, 15, 5.0, testdata=test_data)