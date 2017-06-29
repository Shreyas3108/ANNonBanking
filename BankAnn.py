# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 00:12:57 2017

@author: shrey
"""
import numpy as np 
import pandas as pd 
data = pd.read_csv('Churn_Modelling.csv')
data.head()
x = data.iloc[:,3:13].values
y = data.iloc[:,13].values
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder1 = LabelEncoder()
labelencoder2 = LabelEncoder()
x[:, 1] = labelencoder1.fit_transform(x[:, 1])
x[:, 2]=labelencoder2.fit_transform(x[:,2])
onehotencoder = OneHotEncoder(categorical_features = [1])
x = onehotencoder.fit_transform(x).toarray()
x = x[:, 1:]
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.15,random_state=2)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
from keras.models import Sequential
from keras.layers import Dense
classifier = Sequential()
classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu',input_dim = 11))
classifier.add(Dense(units=6,kernel_initializer='uniform',activation='relu'))
classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
classifier.fit(x_train,y_train,batch_size=10,epochs=100)

y_pred = classifier.predict(x_test)
y_stat = (y_pred>0.5)
from sklearn.metrics import confusion_matrix
res = confusion_matrix(y_test,y_stat)
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test,y_stat)
print(acc*100)
