# -*- coding: utf-8 -*-
"""
Created on Sun Dec 29 12:13:32 2019

@author: MuhammadSalman
"""
from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt

lr = linear_model.LinearRegression()
diabetes = datasets.load_diabetes()
y = diabetes.target

predicted = cross_val_predict(lr, diabetes.data, y, cv=20)

fig, ax = plt.subplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
plt.show()