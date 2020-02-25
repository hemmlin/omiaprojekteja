#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 16:26:23 2020

@author: linda
"""
# source:
# http://neuralnetworksanddeeplearning.com/chap1.html
import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#from datetime import datetime
import sys
import random
#%matplotlib inline

script = sys.argv[0]
print(script)

#tree layer network: 764-> 15 ->10
bin(0)
#%%

#Pixels are organized row-wise. 
#Pixel values are 0 to 255. 0 means background (white), 
#255 means foreground (black). 

 #sigmoid-funtion       
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

#network of 3 layers sizes=[764,15,10]
class Network(object):
    def __init__(self,sizes):
        self.num_layers=len(sizes)
        self.sizes=sizes
        #random generated biases (not for input layer)
        self.biases=[np.random.randn(y,1) for y in sizes[1:]]
        self.weights=[np.random.randn(y,x)
                        for x,y in zip(sizes[:-1],sizes[1:])]
        #feed forvard

    def feedforward(self,a):
        for b,w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a)+b)
        return a
   
        # stochastic gradient descent
        #epoch: times the training data can be used
        #data is list of tuples (x,y) x being input and y the desired output

    def SGD(self, data, epochs, subset_size, eta, testdata=None):
        data=list(data)
        n=len(data)
        #testdata=list(testdata)
        if testdata: 
            testdata=list(testdata)
            n_test= len(testdata)
            
        for j in range(epochs):
            random.shuffle(data)
            subsets=[
                    data[k:k+subset_size]
                    for k in range(0,n,subset_size)]
            for subset in subsets:
                self.update_subset(subset,eta)
            if testdata:
                print("Epoch {}: {} / {}".format(j, self.evaluate(testdata), n_test))
            else:
                print("Epoch {} completet".format(j))

                

    #updating the weights and biases with given subset of training data            
    def update_subset(self, miniset,eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in miniset:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            #summat gradienttien suuntaan
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        #liikutaan gradienttien keskiarvoa vastaan etan verran
        self.weights = [w-(eta/len(miniset))*nw 
                    for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(miniset))*nb 
                   for b, nb in zip(self.biases, nabla_b)]
   
  
    #backprop palauttaa gradientin inputille cost function mukaan
    def backprop(self, dat, goal):  
        nabla_b= [np.zeros(b.shape) for b in self.biases]
        nabla_w=[np.zeros(w.shape) for w in self.weights]
        activation= dat
        activations=[dat] #sailo joka levelin aktivaatioille
        zs=[]
        for b, w in zip(self.biases, self.weights):
            z=np.dot(w,activation)+b
            zs.append(z)
            activation= sigmoid(z)
            activations.append(activation)
            
        delta= self.cost_derivative(activations[-1],goal) *sigmoid_prime(zs[-1])
        nabla_b[-1]=delta
        nabla_w[-1]=np.dot(delta, activations[-2].transpose())
        
        for l in range(2, self.num_layers):
            z=zs[-l]
            sp =sigmoid_prime(z)
            delta=np.dot(self.weights[-l+1].transpose(),delta)*sp
            nabla_b[-l]=delta
            nabla_w[-l]=np.dot(delta,activations[-l-1].transpose())
        return(nabla_b,nabla_w)
    
    
    #backprop palauttaa gradientin inputille cost function mukaan
    def backprop_eff(self, dat, goal):  
        nabla_b= [np.zeros(b.shape) for b in self.biases]
        nabla_w=[np.zeros(w.shape) for w in self.weights]
        activation= dat
        activations=[dat] #sailo joka levelin aktivaatioille
        zs=[]
        for b, w in zip(self.biases, self.weights):
            z=np.dot(w,activation)+b
            zs.append(z)
            activation= sigmoid(z)
            activations.append(activation)
            
        delta= self.cost_derivative(activations[-1],goal) *sigmoid_prime(zs[-1])
        nabla_b[-1]=delta
        nabla_w[-1]=np.dot(delta, activations[-2].transpose())
        
        for l in range(2, self.num_layers):
            z=zs[-l]
            sp =sigmoid_prime(z)
            delta=np.dot(self.weights[-l+1].transpose(),delta)*sp
            nabla_b[-l]=delta
            nabla_w[-l]=np.dot(delta,activations[-l-1].transpose())
        return(nabla_b,nabla_w)

     
    def evaluate(self,tdata):
        test_result= [(np.argmax(self.feedforward(x)),y)
                        for (x,y) in tdata]
        return sum(int(x==y) for (x,y) in test_result)
  
    def cost_derivative(self, output_activations, y):
        return (output_activations-y)
                
            
                
            


















    


