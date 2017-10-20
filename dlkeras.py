from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.cross_validation import *

# Import libraries necessary for this project
import numpy as np
import pandas as pd
from sklearn.cross_validation import ShuffleSplit
from sklearn import metrics
import matplotlib.pyplot as plt
import visuals as vs

data = pd.read_csv('prsa.csv')
data = data.drop('No', axis=1)

data = data.dropna(axis=0, how='any')
pm25 = data["pm2.5"]
#print data["pm2.5"]
data['cbwd'].replace('cv', 0,inplace=True)
data['cbwd'].replace('NW', 1,inplace=True)
data['cbwd'].replace('NE', 2,inplace=True)
data['cbwd'].replace('SW', 3,inplace=True)
data['cbwd'].replace('SE', 4,inplace=True)

#print data.describe()

print "len: ", len(data)

features = data.drop('pm2.5', axis = 1)

X_train, X_test, y_train, y_test = train_test_split(features, pm25, test_size=0.2, random_state=42)

model = Sequential()

model.add(Dense(150, input_dim=11, init='uniform', activation='relu')) #<- 12 is0.011973, 150 is 0.11973
model.add(Dense(50, init='uniform', activation='relu')) #<- 8 is 0.11973, 50 is also 0.011973
model.add(Dense(1, init='uniform', activation='sigmoid'))

model.compile(loss='binary_crossentropy' , optimizer='adam', metrics=['accuracy'])

model.fit(X_train.values, y_train.values, nb_epoch=50, batch_size=len(data)) #<-150 vs 50 no difference

loss_and_metrics = model.evaluate(X_test.values, y_test.values, batch_size=100) #<- adjust doesnt make change

print("\n%s: %f%%" % (model.metrics_names[1], loss_and_metrics[1]*100))
