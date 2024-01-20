#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 13:40:41 2024

@author: pulsaragunawardhana
"""

import math 
import numpy as np 
import pandas as pd
from sklearn import model_selection

class NBC: 
    def _init_(self): 
        self.prior = None
        self.means = None
        self.stds = None
        
    def fit(self, X_train,y_train):
        
        self.unique_labels = y_train.unique()
        
        self.means = np.zeros((len(self.unique_labels),X_train.shape[1]))
        self.stds = np.zeros_like(self.means)
        self.prior = np.zeros(len(self.unique_labels))
        
        print("pass_in func")
        
        for i, label in enumerate(self.unique_labels): 
            X_with_label = X_train[y_train == label]
            print("pass_in func_2")
            #probability for belonging to one class 
            self.prior[i] = len(X_with_label) / len(X_train)
            print("pass_in func_3")
            #For each class and feature caculate the std and mean 
            self.means[i,:] = X_with_label.mean(axis=0)
            print("pass_in func_4")
            self.stds[i,:] = X_with_label.std(axis=0)
            
    def gaussian(self,x,mean,std): 
        return np.prod(( 1/ (np.sqrt(2* math.pi)*std)) * np.exp(-(x-mean)**2 / (2*std**2)),axis=1)
        print("Pass")
    
    def predict(self, X_test): 
        # for all values of x check which p[Y|X] is the maximum 
        probs = np.zeros((len(X_test), len(self.unique_labels)))
        
        for i, label in enumerate(self.unique_labels):
            print("Pass")
            prob_i = self.gaussian(X_test, self.means[i,:], self.stds[i,:]) * self.prior[i]
            probs[:,i] = prob_i
        
        predicted_y = self.unique_labels[np.argmax(probs,axis=1)]
        
        return predicted_y
    
    def get_accuracy(self,y,y_preds):
        correct = y == y_preds 
        acc = correct.sum() / len(y) *100.0 
        acc = float("{:.2f}".format(acc))
        return acc 
    
df = pd.read_csv("diabetes.csv")
X = df.iloc[:,:-1]
Y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,Y,test_size = 0.20, random_state=42)

naive_bayes = NBC()
print('pass')
naive_bayes.fit(X_train, y_train)
print('pass')
preds = naive_bayes.predict(X_test)

acc = naive_bayes.get_accuracy(y_test, preds)
print(acc,'%') 


